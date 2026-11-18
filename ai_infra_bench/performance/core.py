import asyncio
import codecs
import json
import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, AsyncIterator, Dict, Mapping, Optional

import aiohttp

from ai_infra_bench.performance.struct import OutputMetric, TextType
from ai_infra_bench.utils.req import (
    STREAM_RETURN_PAYLOAD,
    extract_error_message,
    is_failed_finish_reason,
)

if TYPE_CHECKING:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

FLUSH_CACHE_TRIES = 10


async def _iter_sse_events(
    response: aiohttp.ClientResponse,
    raw: bool = False,
    non_sse_sample: list[str] | None = None,
) -> AsyncIterator[tuple[str | None, str]]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    current_event: str | None = None

    def process_line(line: str) -> tuple[str | None, str] | None:
        nonlocal current_event
        line = line.strip()
        if not line:
            current_event = None
            return None
        if line.startswith(":"):
            return None
        if raw:
            print(line)
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip() or None
            return None
        if line.startswith("data:"):
            return current_event, line[len("data:") :].lstrip()
        if line.startswith(("id:", "retry:")):
            return None
        if non_sse_sample is not None and len(non_sse_sample) < 10:
            non_sse_sample.append(line)
        return None

    async for chunk_bytes in response.content.iter_any():
        buffer += decoder.decode(chunk_bytes)
        while "\n" in buffer:
            raw_line, buffer = buffer.split("\n", 1)
            if item := process_line(raw_line):
                yield item

    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        if item := process_line(buffer):
            yield item


async def iter_sse_data(
    response: aiohttp.ClientResponse, raw: bool = False
) -> AsyncIterator[str]:
    async for _, data_str in _iter_sse_events(response, raw=raw):
        yield data_str


def _extract_sse_error_event(chunk: str) -> str:
    try:
        parsed = json.loads(chunk)
    except json.JSONDecodeError:
        return chunk
    if err_msg := extract_error_message(parsed):
        return err_msg
    if isinstance(parsed, Mapping) and parsed.get("message"):
        return str(parsed["message"])
    return chunk


def _summarize_non_sse_body(non_sse_lines: list[str]) -> str:
    raw_sample = "\n".join(non_sse_lines).strip()
    if not raw_sample:
        return "Empty streaming response (no SSE data received)"
    try:
        parsed = json.loads(raw_sample)
    except json.JSONDecodeError:
        return f"Unexpected non-SSE HTTP 200 response: {raw_sample[:200]}"
    if err_msg := extract_error_message(parsed):
        return err_msg
    return f"Unexpected non-SSE HTTP 200 response: {raw_sample[:200]}"


def _fail_stream_output(
    output: OutputMetric, start_time: float, request_url: str, message: str
) -> OutputMetric:
    output.latency_ms = (
        (time.perf_counter() - start_time) * 1000 if start_time > 0.0 else 0.0
    )
    output.error_message = message
    output.success = False
    logger.error("Stream error from %s: %s", request_url, message)
    return output


async def request_func(
    session: aiohttp.ClientSession,
    request_url: str,
    payload: Dict,
    headers: Dict | None = None,
    raw: bool = False,
    render_content: bool = False,
    sem: Optional[asyncio.Semaphore] = None,
    pbar: "Optional[tqdm]" = None,
) -> OutputMetric:
    render_content = render_content and not raw
    output = OutputMetric(payload=payload)
    st = 0.0

    try:
        async with sem or nullcontext():
            payload = deepcopy(payload)
            payload.pop("return_meta_info", None)
            payload.update(STREAM_RETURN_PAYLOAD)
            output.payload = payload
            st = time.perf_counter()
            async with session.post(
                url=request_url, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    saw_sse_data = False
                    saw_done = False
                    saw_choices = False
                    non_sse_lines: list[str] = []

                    async for event_type, chunk in _iter_sse_events(
                        response, raw=raw, non_sse_sample=non_sse_lines
                    ):
                        saw_sse_data = True
                        if chunk == "[DONE]":
                            saw_done = True
                            continue

                        if event_type == "error":
                            return _fail_stream_output(
                                output,
                                st,
                                request_url,
                                _extract_sse_error_event(chunk),
                            )

                        data = json.loads(chunk)
                        if not isinstance(data, Mapping):
                            return _fail_stream_output(
                                output,
                                st,
                                request_url,
                                f"Invalid SSE data payload: {chunk[:200]}",
                            )
                        if err_msg := extract_error_message(data):
                            return _fail_stream_output(output, st, request_url, err_msg)

                        output.update_response_metrics(data)

                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        saw_choices = True

                        choice = choices[0]
                        if finish_reason := choice.get("finish_reason"):
                            output.finish_reason = finish_reason

                        # The legacy completions API streams generated text at
                        # choices[].text instead of choices[].delta.content.
                        if "text" in choice:
                            output.update_stream_output(
                                choice.get("text", ""),
                                st,
                                TextType.CONTENT,
                                render_content,
                            )
                            continue

                        # Reasoning models stream thoughts via `reasoning_content`;
                        # count them like content.
                        delta = choice.get("delta") or {}
                        output.update_stream_output(
                            delta.get("reasoning_content", "")
                            or delta.get("reasoning", ""),
                            st,
                            TextType.REASONING,
                            render_content,
                        )
                        output.update_stream_output(
                            delta.get("content", ""),
                            st,
                            TextType.CONTENT,
                            render_content,
                        )

                        if tool_calls := delta.get("tool_calls"):
                            output.update_stream_output(
                                tool_calls, st, TextType.TOOL_CALLS, render_content
                            )

                    if not saw_sse_data:
                        return _fail_stream_output(
                            output,
                            st,
                            request_url,
                            _summarize_non_sse_body(non_sse_lines),
                        )
                    if not saw_choices:
                        return _fail_stream_output(
                            output,
                            st,
                            request_url,
                            "Streaming response contained no choices",
                        )
                    if is_failed_finish_reason(output.finish_reason):
                        return _fail_stream_output(
                            output,
                            st,
                            request_url,
                            f"Generation aborted (finish_reason={output.finish_reason})",
                        )
                    if not saw_done and output.finish_reason is None:
                        return _fail_stream_output(
                            output,
                            st,
                            request_url,
                            "Streaming response ended prematurely before [DONE] or finish_reason",
                        )

                    output.latency_ms = (time.perf_counter() - st) * 1000
                    output.success = True
                else:
                    output.latency_ms = (time.perf_counter() - st) * 1000
                    raw_body = (await response.text()).strip()
                    output.error_message = f"Status Code={getattr(response, 'status', 'N/A')}, Reason: {raw_body}"
                    logger.error("Request Error, %s", output.error_message)
                    output.success = False
    except Exception as error:
        detail = (
            f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        )
        logger.error("Request Error: %s", detail)
        logger.debug("Request exception details", exc_info=True)
        output.error_message = detail
        if st > 0.0:
            output.latency_ms = (time.perf_counter() - st) * 1000
        output.success = False
    finally:
        if pbar:
            pbar.update(1)

    return output


async def flush_cache(session: aiohttp.ClientSession, flush_cache_endpoint: str):
    successful_ct = 0
    for i in range(FLUSH_CACHE_TRIES):
        try:
            await asyncio.sleep(0.2)
            async with session.post(flush_cache_endpoint) as res:
                response_text = await res.text()
                if res.status in (404, 405):
                    logger.warning(
                        f"Cache flush endpoint {flush_cache_endpoint} returned "
                        f"HTTP {res.status}; skipping"
                    )
                    return
                if res.status != 200:
                    logger.warning(
                        f"Failed to send a flush_cache request in {i+1}/{FLUSH_CACHE_TRIES} tries. "
                        f"Error Message: {response_text}"
                    )
                else:
                    successful_ct += 1
        except Exception as error:
            logger.error(
                "Failed to send a flush_cache request in %s/%s tries: %s",
                i + 1,
                FLUSH_CACHE_TRIES,
                error,
            )
    logger.info(
        f"Successfully flush cache in count {successful_ct}/{FLUSH_CACHE_TRIES} tries"
    )
