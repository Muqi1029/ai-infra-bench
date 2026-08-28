"""Replay request sessions with session-level concurrency control.

Each JSONL file matched by ``--payload-regex-path`` represents one session.
Requests within a session are sent in order, while different sessions may run
concurrently. A semaphore token is held for the lifetime of a session so the
configured concurrency describes active sessions rather than individual HTTP
requests.
"""

import asyncio
import json
import logging
import time
from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from tqdm import tqdm

from ai_infra_bench.performance.bench_utils import (
    get_request,
    handle_outputs,
    set_seed,
    wait_for_request_interval,
)
from ai_infra_bench.performance.core import flush_cache, request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import (
    add_common_args,
    api_url,
    parse_override_payload,
    prepare_payload,
)

logger = logging.getLogger(__name__)


def read_session_requests(payload_regex_path: str) -> List[List[Dict[str, Any]]]:
    """Read one ordered request list from each JSONL file matching a glob."""
    paths = sorted(glob(payload_regex_path, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No session files matched --payload-regex-path={payload_regex_path!r}"
        )

    sessions: List[List[Dict[str, Any]]] = []
    request_count = 0
    for file_path in paths:
        logger.info("Reading session %s", file_path)
        session: List[Dict[str, Any]] = []
        with Path(file_path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{file_path}:{line_number}: invalid JSON ({error.msg})"
                    ) from error
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"{file_path}:{line_number}: request must be a JSON object"
                    )
                session.append(dict(payload))
                request_count += 1
        sessions.append(session)

    logger.info("Read %d sessions with %d requests", len(sessions), request_count)
    return sessions


async def request_func_wrapper(
    args: Namespace,
    session,
    request_url: str,
    session_payloads: Iterable[Mapping[str, Any]],
    semaphore: asyncio.Semaphore,
    pbar: tqdm | None = None,
) -> List[OutputMetric]:
    """Run one session while holding exactly one semaphore slot."""
    outputs: List[OutputMetric] = []
    payloads = list(session_payloads)
    request_rate = getattr(args, "request_rate", float("inf"))
    model = getattr(args, "model", None)
    override_payload = getattr(args, "override_payload", None)

    async with semaphore:
        for index, raw_payload in enumerate(payloads):
            payload = prepare_payload(raw_payload, model, override_payload)
            # The session semaphore is already held. Passing it to request_func
            # would acquire the same slot twice and deadlock at concurrency 1.
            outputs.append(await request_func(session, request_url, payload))
            if index + 1 < len(payloads):
                await wait_for_request_interval(request_rate)

    if pbar is not None:
        pbar.update(1)
    return outputs


def flatten_outputs(outputs: Iterable[Iterable[OutputMetric]]) -> List[OutputMetric]:
    """Flatten per-session output lists for the shared metrics renderer."""
    return [output for session_outputs in outputs for output in session_outputs]


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        prog="aib session-bench",
        description="Replay JSONL request sessions with session-level concurrency",
    )
    add_common_args(parser)
    parser.add_argument(
        "payload_regex_path",
        nargs="?",
        help="Glob matching JSONL session files (one request per line)",
    )
    parser.add_argument(
        "--payload-regex-path",
        dest="payload_regex_path_option",
        help="Glob matching JSONL session files",
    )
    parser.add_argument("--num-sessions", type=int, help="Maximum sessions to run")
    parser.add_argument(
        "--num-warmup-sessions",
        type=int,
        default=3,
        help="Sessions to run and discard before the measured phase",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum active sessions",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Session start/request rate; defaults to unlimited",
    )
    parser.add_argument("--dump-path", help="Optional JSONL output dump path")
    parser.add_argument(
        "--dump-content",
        choices=["all", "msg"],
        default="all",
        help="Output dump format",
    )
    parser.add_argument("--metric-path", help="Optional JSON/JSONL metrics path")
    parser.add_argument("--label", help="Label for exported metrics")
    parser.add_argument(
        "--disable-flush-cache",
        action="store_true",
        help="Skip the cache flush between warmup and measured sessions",
    )
    parser.add_argument("--debug", action="store_true", help="Run at most 3 sessions")
    args = parser.parse_args(argv)
    args.payload_regex_path = args.payload_regex_path_option or args.payload_regex_path
    if not args.payload_regex_path:
        parser.error("a session payload glob is required")
    return args


def validate_args(args: Namespace) -> None:
    if args.override_payload:
        parse_override_payload(args.override_payload)
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be > 0")
    if args.num_warmup_sessions < 0:
        raise ValueError("--num-warmup-sessions must be >= 0")
    if args.num_sessions is not None and args.num_sessions < 1:
        raise ValueError("--num-sessions must be >= 1")
    if args.metric_path and not args.metric_path.lower().endswith((".json", ".jsonl")):
        raise ValueError("--metric-path must end with .json or .jsonl")
    if args.dump_path and not args.dump_path.lower().endswith(".jsonl"):
        args.dump_path = f"{args.dump_path}.jsonl"


async def run_benchmark(args: Namespace) -> None:
    session_requests = read_session_requests(args.payload_regex_path)
    if args.debug:
        args.num_sessions = 3
    if args.num_sessions is not None:
        session_requests = session_requests[: args.num_sessions]
    logger.info("Running %d sessions", len(session_requests))

    warmup = session_requests[: args.num_warmup_sessions]
    formal = session_requests[args.num_warmup_sessions :]
    request_url = api_url(args.base_url, "/v1/chat/completions")
    flush_cache_endpoint = api_url(args.base_url, "/flush_cache")
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async with _create_bench_client_session(
        args.max_concurrency, args.api_key
    ) as client:
        if warmup:
            logger.info("Warming up %d sessions", len(warmup))
            with tqdm(total=len(warmup), desc="Warmup sessions") as pbar:
                await asyncio.gather(
                    *(
                        request_func_wrapper(
                            args, client, request_url, payloads, semaphore, pbar
                        )
                        for payloads in warmup
                    )
                )

        if not args.disable_flush_cache:
            await flush_cache(client, flush_cache_endpoint)

        outputs_by_session: List[List[OutputMetric]] = []
        with tqdm(total=len(formal), desc="Formally running sessions") as pbar:
            start = time.perf_counter()
            tasks = []
            async for payloads in get_request(formal, args.request_rate):
                tasks.append(
                    asyncio.create_task(
                        request_func_wrapper(
                            args, client, request_url, payloads, semaphore, pbar
                        )
                    )
                )
            if tasks:
                outputs_by_session = await asyncio.gather(*tasks)
            duration_s = max(time.perf_counter() - start, 1e-9)

    handle_outputs(
        outputs=flatten_outputs(outputs_by_session),
        duration_s=duration_s,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        dump_path=args.dump_path,
        dump_content=args.dump_content,
        metric_path=args.metric_path,
        label=args.label,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        set_seed(args.seed)
        asyncio.run(run_benchmark(args))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser = ArgumentParser(prog="aib session-bench")
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
