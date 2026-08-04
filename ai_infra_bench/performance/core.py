import asyncio
import codecs
import json
import logging
import sys
import time
import traceback
from argparse import Namespace
from contextlib import nullcontext
from typing import AsyncIterator, Dict, Optional

import aiohttp
from tqdm import tqdm

from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.req import STREAM_RETURN_PAYLOAD

logger = logging.getLogger(__name__)


async def iter_sse_data(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""

    async for chunk_bytes in response.content.iter_any():
        buffer += decoder.decode(chunk_bytes)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[len("data:") :].lstrip()

    buffer += decoder.decode(b"", final=True)
    line = buffer.strip()
    if line and line.startswith("data:"):
        yield line[len("data:") :].lstrip()


async def request_func(
    session: aiohttp.ClientSession,
    request_url: str,
    payload: Dict,
    sem: Optional[asyncio.Semaphore] = None,
    pbar: Optional[tqdm] = None,
):
    payload.update(STREAM_RETURN_PAYLOAD)

    output = OutputMetric(payload=payload)
    st = 0.0

    try:
        async with sem or nullcontext():
            st = time.perf_counter()
            async with session.post(url=request_url, json=payload) as response:
                if response.status == 200:
                    async for chunk in iter_sse_data(response):
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

                        # Reasoning models stream thoughts via `reasoning_content`;
                        # count them like content.
                        delta = choice.get("delta") or {}
                        output.update_stream_output(
                            delta.get("reasoning_content", ""),
                            st,
                        )
                        output.update_stream_output(
                            delta.get("content", ""),
                            st,
                        )

                        if tool_calls := delta.get("tool_calls"):
                            tool_text_parts = []
                            for tool_call in tool_calls:
                                function = tool_call.get("function") or {}
                                if func_name := function.get("name"):
                                    tool_text_parts.append(
                                        "\n\n[Tool Call Detected]: "
                                        f"Function={func_name}\nArgument:"
                                    )
                                if func_arg := function.get("arguments"):
                                    tool_text_parts.append(func_arg)
                            output.update_stream_output(
                                "".join(tool_text_parts),
                                st,
                            )

                    output.latency_ms = (time.perf_counter() - st) * 1000
                    output.success = True
                else:
                    output.latency_ms = (time.perf_counter() - st) * 1000
                    output.error_message = await response.text()
                    print(output.error_message)
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
