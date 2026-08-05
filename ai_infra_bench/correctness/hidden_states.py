"""Inspect and compare hidden states from chat-completion endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import sanitize_url

logger = logging.getLogger(__name__)

PROMPT = "Hello, ai-infra-bench! I am Muqi Li, who are you?"
PAYLOAD = {
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "max_tokens": 1,
    "return_hidden_states": True,
}

_USE_COLOR = False


@dataclass(frozen=True)
class HiddenStateResult:
    base_url: str
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)


@dataclass(frozen=True)
class ComparisonResult:
    reference: HiddenStateResult
    target: HiddenStateResult
    shape_match: bool
    passed: bool
    total_values: int = 0
    differing_values: int = 0
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    rmse: Optional[float] = None
    cosine_similarity: Optional[float] = None
    max_diff_index: Optional[tuple[int, ...]] = None
    reference_at_max: Optional[float] = None
    target_at_max: Optional[float] = None


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return result


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-urls",
        nargs="+",
        required=True,
        type=sanitize_url,
        help="OpenAI-compatible base URLs; the first endpoint is the reference",
    )
    parser.add_argument(
        "--rtol",
        type=_nonnegative_float,
        default=1e-2,
        help="Relative tolerance for hidden-state comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--atol",
        type=_nonnegative_float,
        default=1e-2,
        help="Absolute tolerance for hidden-state comparison (default: %(default)s)",
    )
    parser.add_argument("--prompt", "--user-prompt", default=PROMPT, help="User prompt")
    parser.add_argument(
        "--max-tokens",
        "--max-completion-tokens",
        type=_positive_int,
        default=PAYLOAD["max_tokens"],
        help="Maximum output tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--override-payload",
        type=_json_object,
        metavar="JSON",
        help="JSON object merged into the request payload",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="Bearer token (default: OPENAI_API_KEY or EMPTY)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colors in terminal output"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log raw responses"
    )
    return parser.parse_args(args)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        **PAYLOAD,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    if args.override_payload:
        payload.update(args.override_payload)
    if payload.get("stream"):
        raise ValueError("streaming responses are not supported; set stream to false")
    return payload


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
    session = _create_bench_client_session(
        max_concurrency=max(1, len(base_urls)), api_key=api_key
    )
    async with session:
        requests = [
            _request_completion(session, base_url, payload) for base_url in base_urls
        ]
        return await asyncio.gather(*requests)


def extract_hidden_states(base_url: str, response: dict[str, Any]) -> HiddenStateResult:
    """Validate and normalize hidden states from a completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError(f"{base_url} response has no choices[0]")

    choice = choices[0]
    hidden_states = choice.get("hidden_states")
    if hidden_states is None:
        message = choice.get("message")
        if isinstance(message, dict):
            hidden_states = message.get("hidden_states")
    if hidden_states is None:
        raise ValueError(
            f"{base_url} returned no hidden states; "
            "ensure return_hidden_states=true is supported"
        )

    try:
        values = np.asarray(hidden_states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{base_url} returned invalid hidden states") from exc
    if values.size == 0:
        raise ValueError(f"{base_url} returned empty hidden states")
    return HiddenStateResult(base_url, values)


def compare_results(
    reference: HiddenStateResult,
    target: HiddenStateResult,
    rtol: float,
    atol: float,
) -> ComparisonResult:
    if reference.shape != target.shape:
        return ComparisonResult(reference, target, shape_match=False, passed=False)

    difference = np.abs(reference.values - target.values)
    close = np.isclose(reference.values, target.values, rtol=rtol, atol=atol)
    differing_values = int(close.size - np.count_nonzero(close))
    max_flat_index = int(np.argmax(difference))
    max_diff_index = tuple(
        int(index) for index in np.unravel_index(max_flat_index, difference.shape)
    )

    reference_flat = reference.values.ravel()
    target_flat = target.values.ravel()
    reference_norm = float(np.linalg.norm(reference_flat))
    target_norm = float(np.linalg.norm(target_flat))
    norm_product = reference_norm * target_norm
    if norm_product:
        cosine_similarity = float(np.dot(reference_flat, target_flat) / norm_product)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    else:
        cosine_similarity = 1.0 if reference_norm == target_norm else 0.0

    return ComparisonResult(
        reference=reference,
        target=target,
        shape_match=True,
        passed=differing_values == 0,
        total_values=int(close.size),
        differing_values=differing_values,
        max_abs_diff=float(np.max(difference)),
        mean_abs_diff=float(np.mean(difference)),
        rmse=float(np.sqrt(np.mean(np.square(reference.values - target.values)))),
        cosine_similarity=cosine_similarity,
        max_diff_index=max_diff_index,
        reference_at_max=float(reference.values[max_diff_index]),
        target_at_max=float(target.values[max_diff_index]),
    )


def _style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _status(passed: bool) -> str:
    return _style("✓ PASS", "32;1") if passed else _style("✗ FAIL", "31;1")


def _section(title: str) -> None:
    print(f"\n{_style(title, '36;1')}\n{'-' * len(title)}")


def _number(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.8e}"


def _shape(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(size) for size in shape) + "]"


def print_endpoint_summary(
    results: Sequence[HiddenStateResult], comparisons: Sequence[ComparisonResult]
) -> None:
    _section("Endpoint summary")
    for index, result in enumerate(results):
        status = (
            _style("● REF", "36;1")
            if index == 0
            else _status(comparisons[index - 1].passed)
        )
        print(f"  {status}  #{index + 1} {result.base_url}")
        print(f"       shape={_shape(result.shape)}, values={result.values.size}")


def print_single_result(result: HiddenStateResult) -> None:
    _section("Hidden states")
    print(f"  Shape: {_shape(result.shape)}")
    print(f"  Values: {result.values.size}")
    print(f"  Min: {_number(float(np.min(result.values)))}")
    print(f"  Max: {_number(float(np.max(result.values)))}")
    print(f"  Mean: {_number(float(np.mean(result.values)))}")
    print(f"  L2 norm: {_number(float(np.linalg.norm(result.values)))}")


def print_comparison(comparison: ComparisonResult, index: int) -> None:
    _section(f"Comparison #{index}: reference → {comparison.target.base_url}")
    print(
        f"  {_status(comparison.shape_match)}  Shape: "
        f"{_shape(comparison.reference.shape)} vs {_shape(comparison.target.shape)}"
    )
    if not comparison.shape_match:
        print(f"  {_status(False)}  Overall: shapes must match")
        return

    close_values = comparison.total_values - comparison.differing_values
    print(
        f"  {_status(comparison.passed)}  Values within tolerance: "
        f"{close_values}/{comparison.total_values}"
    )
    print(f"  Max |Δ|: {_number(comparison.max_abs_diff)}")
    print(f"  Mean |Δ|: {_number(comparison.mean_abs_diff)}")
    print(f"  RMSE: {_number(comparison.rmse)}")
    print(f"  Cosine similarity: {_number(comparison.cosine_similarity)}")
    print(
        f"  Largest difference at {comparison.max_diff_index}: "
        f"reference={_number(comparison.reference_at_max)}, "
        f"target={_number(comparison.target_at_max)}"
    )
    print(f"  {_status(comparison.passed)}  Overall: all values must be close")


async def run(args: argparse.Namespace) -> bool:
    payload = build_payload(args)
    responses = await fetch_chat_completions(args.base_urls, payload, args.api_key)
    results = [
        extract_hidden_states(base_url, response)
        for base_url, response in zip(args.base_urls, responses)
    ]
    comparisons = [
        compare_results(results[0], result, args.rtol, args.atol)
        for result in results[1:]
    ]

    print(_style("\nHIDDEN STATES INSPECTION", "36;1"))
    print(
        f"Reference: {results[0].base_url}  │  rtol={args.rtol:g}  │  "
        f"atol={args.atol:g}"
    )
    print_endpoint_summary(results, comparisons)
    if len(results) == 1:
        print_single_result(results[0])
        return True

    for index, comparison in enumerate(comparisons, start=1):
        print_comparison(comparison, index)

    passed = all(comparison.passed for comparison in comparisons)
    _section("Final result")
    message = (
        "All endpoints match the reference."
        if passed
        else "Differences detected; see failed checks above."
    )
    print(f"{_status(passed)}  {message}")
    return passed


def main(args: Optional[Sequence[str]] = None) -> int:
    global _USE_COLOR
    parsed_args = parse_args(args)
    _USE_COLOR = (
        not parsed_args.no_color
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
    )
    if parsed_args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    try:
        passed = asyncio.run(run(parsed_args))
    except Exception as exc:
        print(_style(f"ERROR: {exc}", "31;1"), file=sys.stderr)
        logger.debug("hidden-state inspection failed", exc_info=True)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
