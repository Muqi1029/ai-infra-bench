import asyncio
import codecs
import json
import logging
import sys
import time
import traceback
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp
from tqdm import tqdm

from ai_infra_bench.performance.struct import OutputMetric, TextType
from ai_infra_bench.utils.req import STREAM_RETURN_PAYLOAD

logger = logging.getLogger(__name__)

FLUSH_CACHE_TRIES = 10


async def iter_sse_data(
    response: aiohttp.ClientResponse, raw: bool = False
) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    async for chunk_bytes in response.content.iter_any():
        buffer += decoder.decode(chunk_bytes)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if raw:
                print(line)
            if line.startswith("data:"):
                yield line[len("data:") :].lstrip()

    buffer += decoder.decode(b"", final=True)
    line = buffer.strip()
    if raw:
        print(line)
    if line and line.startswith("data:"):
        yield line[len("data:") :].lstrip()


async def request_func(
    session: aiohttp.ClientSession,
    request_url: str,
    payload: Dict,
    headers: Dict | None = None,
    raw: bool = False,
    render_content: bool = False,
    sem: Optional[asyncio.Semaphore] = None,
    pbar: Optional[tqdm] = None,
) -> OutputMetric:
    payload = deepcopy(payload)
    payload.pop("return_meta_info", None)
    payload.update(STREAM_RETURN_PAYLOAD)

    render_content = render_content and not raw
    output = OutputMetric(payload=payload)
    st = 0.0

    try:
        async with sem or nullcontext():
            st = time.perf_counter()
            async with session.post(
                url=request_url, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    async for chunk in iter_sse_data(response, raw=raw):
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)

                        output.update_response_metrics(data)

                        choices = data.get("choices") or []
                        if not choices:
                            continue

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
                            delta.get("reasoning_content", ""),
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

                    output.latency_ms = (time.perf_counter() - st) * 1000
                    output.success = True
                else:
                    output.latency_ms = (time.perf_counter() - st) * 1000
                    output.error_message = await response.text()
                    logger.error(
                        f"Request Error, Status Code="
                        f"{getattr(response, 'status', 'N/A')}, Reason: {output.error_message}",
                    )
                    output.success = False
    except Exception:
        exc_info = sys.exc_info()
        error_message = "".join(traceback.format_exception(*exc_info))
        logger.error(error_message)
        output.error_message = error_message
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
            res = await session.post(flush_cache_endpoint)
            if res.status != 200:
                error_message = await res.text()
                logger.warning(
                    f"Failed to send a flush_cache request in {i+1}/{FLUSH_CACHE_TRIES} tries. "
                    f"Error Message: {error_message}"
                )
            else:
                successful_ct += 1
        except Exception:
            exc_info = sys.exc_info()
            error_message = "".join(traceback.format_exception(*exc_info))
            logger.error(error_message)
    logger.info(
        f"Successfully flush cache in count {successful_ct}/{FLUSH_CACHE_TRIES} tries"
    )
