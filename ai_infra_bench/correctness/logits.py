"""Inspect and compare token log probabilities from chat-completion endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Optional, Sequence

from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import sanitize_url

logger = logging.getLogger(__name__)

PROMPT = "Hello, ai-infra-bench! I am Muqi Li, who are you?"
PAYLOAD = {
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0,
    "max_tokens": 1,
    "logprobs": True,
    "top_logprobs": 3,
}

_USE_COLOR = False


@dataclass(frozen=True)
class TopLogProb:
    token: str
    logprob: float


@dataclass(frozen=True)
class TokenLogProb:
    token: str
    logprob: float
    top_logprobs: tuple[TopLogProb, ...]


@dataclass(frozen=True)
class LogProbResult:
    base_url: str
    tokens: tuple[TokenLogProb, ...]


@dataclass(frozen=True)
class TokenDiff:
    position: int
    reference: Optional[TokenLogProb]
    target: Optional[TokenLogProb]
    rtol: float
    atol: float

    @property
    def token_match(self) -> bool:
        return (
            self.reference is not None
            and self.target is not None
            and self.reference.token == self.target.token
        )

    @property
    def logprob_close(self) -> bool:
        return (
            self.reference is not None
            and self.target is not None
            and _is_close(
                self.reference.logprob,
                self.target.logprob,
                self.rtol,
                self.atol,
            )
        )

    @property
    def candidates(
        self,
    ) -> tuple[tuple[Optional[TopLogProb], Optional[TopLogProb]], ...]:
        reference = self.reference.top_logprobs if self.reference else ()
        target = self.target.top_logprobs if self.target else ()
        return tuple(zip_longest(reference, target))

    @staticmethod
    def candidate_token_match(
        reference: Optional[TopLogProb], target: Optional[TopLogProb]
    ) -> bool:
        return (
            reference is not None
            and target is not None
            and reference.token == target.token
        )

    def candidate_logprob_close(
        self, reference: Optional[TopLogProb], target: Optional[TopLogProb]
    ) -> bool:
        return (
            reference is not None
            and target is not None
            and _is_close(reference.logprob, target.logprob, self.rtol, self.atol)
        )

    def candidate_passed(
        self, reference: Optional[TopLogProb], target: Optional[TopLogProb]
    ) -> bool:
        token_match = self.candidate_token_match(reference, target)
        return token_match and self.candidate_logprob_close(reference, target)

    @property
    def top_tokens_match(self) -> bool:
        return bool(self.candidates) and all(
            self.candidate_token_match(reference, target)
            for reference, target in self.candidates
        )

    @property
    def top_logprobs_close(self) -> bool:
        return bool(self.candidates) and all(
            self.candidate_logprob_close(reference, target)
            for reference, target in self.candidates
        )

    @property
    def top_k_passed(self) -> bool:
        return self.top_tokens_match and self.top_logprobs_close

    def passed(self, compare_top_logprobs: bool) -> bool:
        selected_token_passed = self.token_match and self.logprob_close
        return selected_token_passed and (not compare_top_logprobs or self.top_k_passed)


@dataclass(frozen=True)
class ComparisonResult:
    reference: LogProbResult
    target: LogProbResult
    token_diffs: tuple[TokenDiff, ...]
    compare_top_logprobs: bool

    @property
    def length_match(self) -> bool:
        return len(self.reference.tokens) == len(self.target.tokens)

    @property
    def tokens_match(self) -> bool:
        return self.length_match and all(diff.token_match for diff in self.token_diffs)

    @property
    def selected_logprobs_close(self) -> bool:
        return self.length_match and all(
            diff.logprob_close for diff in self.token_diffs
        )

    @property
    def top_tokens_match(self) -> bool:
        return not self.compare_top_logprobs or (
            self.length_match
            and all(diff.top_tokens_match for diff in self.token_diffs)
        )

    @property
    def top_logprobs_close(self) -> bool:
        return not self.compare_top_logprobs or (
            self.length_match
            and all(diff.top_logprobs_close for diff in self.token_diffs)
        )

    @property
    def top_k_passed(self) -> bool:
        return self.top_tokens_match and self.top_logprobs_close

    @property
    def passed(self) -> bool:
        return self.tokens_match and self.selected_logprobs_close and self.top_k_passed


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


def _top_k(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 20:
        raise argparse.ArgumentTypeError("must be between 1 and 20")
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
        default=0.0,
        help="Relative tolerance for logprob comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--atol",
        type=_nonnegative_float,
        default=1e-2,
        help="Absolute tolerance for logprob comparison (default: %(default)s)",
    )
    parser.add_argument("--prompt", "--user-prompt", default=PROMPT, help="User prompt")
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=PAYLOAD["max_tokens"],
        help="Maximum output tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=_top_k,
        default=PAYLOAD["top_logprobs"],
        help="Number of Top-K candidates to compare (default: %(default)s)",
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
        "top_logprobs": args.top_logprobs,
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


def extract_logprobs(base_url: str, response: dict[str, Any]) -> LogProbResult:
    """Validate and normalize an OpenAI chat-completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError(f"{base_url} response has no choices[0]")

    choice = choices[0]
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content:
        raise ValueError(
            f"{base_url} returned no token logprobs; ensure logprobs=true is supported"
        )

    tokens = []
    for position, item in enumerate(content):
        if not isinstance(item, dict) or "token" not in item or "logprob" not in item:
            raise ValueError(
                f"{base_url} has invalid logprob item at position {position}"
            )
        try:
            candidates = tuple(
                TopLogProb(str(candidate["token"]), float(candidate["logprob"]))
                for candidate in (item.get("top_logprobs") or [])
            )
            tokens.append(
                TokenLogProb(str(item["token"]), float(item["logprob"]), candidates)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{base_url} has invalid numeric logprob data at position {position}"
            ) from exc

    return LogProbResult(base_url, tuple(tokens))


def _is_close(reference: float, target: float, rtol: float, atol: float) -> bool:
    return math.isclose(reference, target, rel_tol=rtol, abs_tol=atol)


def compare_results(
    reference: LogProbResult,
    target: LogProbResult,
    rtol: float,
    atol: float,
    compare_top_logprobs: bool = True,
) -> ComparisonResult:
    token_diffs = []

    for position, (reference_token, target_token) in enumerate(
        zip_longest(reference.tokens, target.tokens)
    ):
        token_diffs.append(
            TokenDiff(position, reference_token, target_token, rtol, atol)
        )
    return ComparisonResult(reference, target, tuple(token_diffs), compare_top_logprobs)


def _style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _status(passed: bool) -> str:
    return _style("✓ PASS", "32;1") if passed else _style("✗ FAIL", "31;1")


def _section(title: str) -> None:
    print(f"\n{_style(title, '36;1')}\n{'-' * len(title)}")


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _number(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.8f}"


def _difference(left: Optional[float], right: Optional[float]) -> str:
    if left is None or right is None:
        return "—"
    return f"{abs(left - right):.3e}"


def _max_selected_diff(comparison: ComparisonResult) -> Optional[float]:
    differences = [
        abs(diff.reference.logprob - diff.target.logprob)
        for diff in comparison.token_diffs
        if diff.reference is not None and diff.target is not None
    ]
    return max(differences, default=None)


def _max_top_diff(comparison: ComparisonResult) -> Optional[float]:
    differences = [
        abs(reference.logprob - target.logprob)
        for diff in comparison.token_diffs
        for reference, target in diff.candidates
        if reference is not None and target is not None
    ]
    return max(differences, default=None)


def _print_check(name: str, passed: bool, details: str) -> None:
    print(f"  {_status(passed)}  {name}: {details}")


def print_endpoint_summary(
    results: Sequence[LogProbResult], comparisons: Sequence[ComparisonResult]
) -> None:
    _section("Endpoint summary")
    for index, result in enumerate(results):
        status = (
            _style("● REF", "36;1")
            if index == 0
            else _status(comparisons[index - 1].passed)
        )
        top_k = max((len(token.top_logprobs) for token in result.tokens), default=0)
        print(f"  {status}  #{index + 1} {result.base_url}")
        print(f"       tokens={len(result.tokens)}, top-k={top_k}")


def print_single_result(result: LogProbResult) -> None:
    _section("Token logprobs")
    for position, token in enumerate(result.tokens):
        print(f"  Position {position}")
        print(
            f"    Selected token: {_quoted(token.token)}, "
            f"logprob={_number(token.logprob)}, "
            f"probability={math.exp(token.logprob):.4%}"
        )
        if token.top_logprobs:
            print("    Top-K candidates (may include the selected token):")
            for rank, candidate in enumerate(token.top_logprobs, 1):
                print(
                    f"      rank={rank}: {_quoted(candidate.token)}, "
                    f"logprob={_number(candidate.logprob)}"
                )


def print_comparison(comparison: ComparisonResult, index: int) -> None:
    reference_count = len(comparison.reference.tokens)
    target_count = len(comparison.target.tokens)
    total_positions = len(comparison.token_diffs)
    matched_tokens = sum(diff.token_match for diff in comparison.token_diffs)

    _section(f"Comparison #{index}: reference → {comparison.target.base_url}")
    _print_check(
        "Token count", comparison.length_match, f"{reference_count} vs {target_count}"
    )
    _print_check(
        "Selected tokens",
        comparison.tokens_match,
        f"{matched_tokens}/{total_positions} positions match",
    )
    _print_check(
        "Selected logprobs",
        comparison.selected_logprobs_close,
        f"max |Δ| = {_difference(0.0, _max_selected_diff(comparison))}",
    )
    if comparison.compare_top_logprobs:
        matched_top_tokens = sum(
            diff.top_tokens_match for diff in comparison.token_diffs
        )
        _print_check(
            "Top-K tokens by rank",
            comparison.top_tokens_match,
            f"{matched_top_tokens}/{total_positions} positions match",
        )
        _print_check(
            "Top-K candidate logprobs",
            comparison.top_logprobs_close,
            f"max |Δ| = {_difference(0.0, _max_top_diff(comparison))}",
        )
        _print_check(
            "Top-K overall",
            comparison.top_k_passed,
            "tokens and logprobs must both pass",
        )
    _print_check("Overall", comparison.passed, "all checks must pass")

    print("\nPer-position details")
    for diff in comparison.token_diffs:
        reference = diff.reference
        target = diff.target
        absolute_diff = _difference(
            reference.logprob if reference else None,
            target.logprob if target else None,
        )
        position_passed = diff.passed(comparison.compare_top_logprobs)
        print(f"\n  Position {diff.position}: {_status(position_passed)}")
        print("    Selected token:")
        print(
            f"      Reference: {_quoted(reference.token) if reference else '—'}, "
            f"logprob={_number(reference.logprob if reference else None)}"
        )
        print(
            f"      Target:    {_quoted(target.token) if target else '—'}, "
            f"logprob={_number(target.logprob if target else None)}"
        )
        print(
            f"      Result: token match={_status(diff.token_match)}, "
            f"logprob close={_status(diff.logprob_close)}, |Δ|={absolute_diff}"
        )

        if comparison.compare_top_logprobs:
            print("    Top-K candidates:")
            if not diff.candidates:
                print("      No candidates were returned.")
            for rank, (reference_candidate, target_candidate) in enumerate(
                diff.candidates, 1
            ):
                absolute_diff = _difference(
                    reference_candidate.logprob if reference_candidate else None,
                    target_candidate.logprob if target_candidate else None,
                )
                reference_token = (
                    _quoted(reference_candidate.token) if reference_candidate else "—"
                )
                target_token = (
                    _quoted(target_candidate.token) if target_candidate else "—"
                )
                reference_logprob = _number(
                    reference_candidate.logprob if reference_candidate else None
                )
                target_logprob = _number(
                    target_candidate.logprob if target_candidate else None
                )
                token_match = diff.candidate_token_match(
                    reference_candidate, target_candidate
                )
                logprob_close = diff.candidate_logprob_close(
                    reference_candidate, target_candidate
                )
                passed = diff.candidate_passed(reference_candidate, target_candidate)
                print(f"      Rank {rank}: {_status(passed)}")
                print(
                    f"        Reference: token={reference_token}, "
                    f"logprob={reference_logprob}"
                )
                print(
                    f"        Target:    token={target_token}, "
                    f"logprob={target_logprob}"
                )
                print(
                    f"        Result: token match={_status(token_match)}, "
                    f"logprob close={_status(logprob_close)}, "
                    f"|Δ|={absolute_diff}"
                )


async def run(args: argparse.Namespace) -> bool:
    payload = build_payload(args)
    responses = await fetch_chat_completions(args.base_urls, payload, args.api_key)
    results = [
        extract_logprobs(base_url, response)
        for base_url, response in zip(args.base_urls, responses)
    ]
    compare_top_logprobs = bool(
        payload.get("logprobs") and payload.get("top_logprobs", 0)
    )
    comparisons = [
        compare_results(results[0], result, args.rtol, args.atol, compare_top_logprobs)
        for result in results[1:]
    ]

    print(_style("\nLOGPROB INSPECTION", "36;1"))
    print(
        f"Reference: {results[0].base_url}  │  rtol={args.rtol:g}  │  "
        f"atol={args.atol:g}  │  Top-K={payload.get('top_logprobs', 0)}"
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
        logger.setLevel(logging.DEBUG)
    try:
        passed = asyncio.run(run(parsed_args))
    except Exception as exc:
        print(_style(f"ERROR: {exc}", "31;1"), file=sys.stderr)
        logger.debug("logprob inspection failed", exc_info=True)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
