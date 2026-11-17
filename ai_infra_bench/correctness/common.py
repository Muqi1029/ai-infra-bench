"""Shared CLI, HTTP, and terminal formatting helpers for correctness inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Hello, ai-infra-bench! I am Muqi Li, who are you?"


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def should_use_color(no_color: bool) -> bool:
    return not no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text: str, code: str, use_color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if use_color else text


def status_badge(passed: bool, use_color: bool) -> str:
    return (
        style("✓ PASS", "32;1", use_color)
        if passed
        else style("✗ FAIL", "31;1", use_color)
    )


def print_section(title: str, use_color: bool) -> None:
    print(f"\n{style(title, '36;1', use_color)}\n{'-' * len(title)}")


def read_prompt_arg(prompt: Optional[str], default: str = DEFAULT_PROMPT) -> str:
    if prompt is None:
        return default
    return sys.stdin.read() if prompt == "-" else prompt


async def _request_completion(
    session: Any, base_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    async with session.post(url, json=payload) as response:
        raw_body = await response.text()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{base_url} returned invalid JSON (HTTP {response.status}): "
                f"{raw_body[:300]!r}"
            ) from exc
        if response.status != 200:
            raise RuntimeError(
                f"{base_url} returned HTTP {response.status}: "
                f"{json.dumps(body, ensure_ascii=False)[:500]}"
            )
        logger.debug("Response from %s:\n%s", base_url, json.dumps(body, indent=2))
        return body


async def fetch_chat_completions(
    base_urls: Sequence[str], payload: dict[str, Any], api_key: str = "EMPTY"
) -> list[dict[str, Any]]:
    from ai_infra_bench.utils.client import _create_bench_client_session

    session = _create_bench_client_session(
        max_concurrency=max(1, len(base_urls)), api_key=api_key
    )
    async with session:
        requests = [
            _request_completion(session, base_url, payload) for base_url in base_urls
        ]
        return await asyncio.gather(*requests)
