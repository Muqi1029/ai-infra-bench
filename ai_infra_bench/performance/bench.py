# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
import asyncio
import json
import logging
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tqdm import tqdm

from ai_infra_bench.performance.bench_utils import get_request, handle_outputs, set_seed
from ai_infra_bench.performance.common_args import add_common_args
from ai_infra_bench.performance.core import request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.io import _read_json, _read_jsonl
from ai_infra_bench.utils.req import api_url, prepare_payload

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def tool_filter_request(request: Mapping[str, Any]) -> bool:
    """Return whether a request can run without constrained decoding."""
    tool_choice = request.get("tool_choice", request.get("tool_choices"))
    if tool_choice == "required" or isinstance(tool_choice, dict):
        return False
    if any(
        tool.get("strict") or (tool.get("function") or {}).get("strict")
        for tool in request.get("tools") or []
    ):
        return False
    return not bool(request.get("response_format"))


def read_requests_with_ts(payload_regex_path: str) -> List[Dict]:
    timestamped_requests = []
    for file_path in sorted(glob(payload_regex_path, recursive=True)):
        logger.info(f"Reading {file_path}")
        timestamped_requests.extend(_read_json(file_path))

    timestamped_requests.sort(key=lambda item: datetime.strptime(item[0], DATE_FORMAT))
    requests = [json.loads(content) for _, content in timestamped_requests]
    logger.info(f"Read {len(requests)} requests")
    return requests


def read_requests(payload_regex_path: str) -> List[Dict]:
    requests = []
    for file_path in sorted(glob(payload_regex_path, recursive=True)):
        logger.info(f"Reading {file_path}")
        if file_path.endswith(".json"):
            requests.extend(_read_json(file_path))
        elif file_path.endswith(".jsonl"):
            requests.extend(_read_jsonl(file_path))
        else:
            logger.error(f"{file_path} cannot be read. Only support json/jsonl suffix")
    logger.info(f"Read {len(requests)} requests")
    return requests


def load_requests(args: Namespace) -> List[Dict]:
    requests = []
    if args.with_ts:
        requests = read_requests_with_ts(args.payload_regex_path)
    else:
        requests = read_requests(args.payload_regex_path)

    if args.filter_constrained_grammar_requests:
        filtered_requests = [
            request for request in requests if tool_filter_request(request)
        ]
        num_filtered_requests = len(requests) - len(filtered_requests)
        logger.info(f"Filter {num_filtered_requests} due to constrained decoding")
        requests = filtered_requests
    return requests


def validate_args(args: Namespace) -> None:
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be > 0")
    if args.num_warmup_requests < 0:
        raise ValueError("--num-warmup-requests must be >= 0")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")


async def run_requests(
    session,
    request_url: str,
    requests: Iterable[Mapping[str, Any]],
    model: str,
    override_payload: str,
    semaphore: asyncio.Semaphore,
    progress: Optional[tqdm],
    request_rate: float = float("inf"),
    benchmark_start_time: Optional[float] = None,
) -> List[OutputMetric]:
    tasks = []
    completion_tokens_sum = 0

    async def run_request(payload: Dict[str, Any]) -> OutputMetric:
        nonlocal completion_tokens_sum

        output = await request_func(
            session,
            request_url,
            payload,
            semaphore,
            None,
        )
        if progress is not None:
            progress.update(1)
            if benchmark_start_time is not None:
                if output.success:
                    completion_tokens_sum += output.completion_tokens
                elapsed_s = max(time.perf_counter() - benchmark_start_time, 1e-9)
                progress.set_postfix(
                    {"TPS": (f"{completion_tokens_sum / elapsed_s:.2f} tokens/s")}
                )
        return output

    async for request in get_request(requests, request_rate):
        payload = prepare_payload(request, model)
        if override_payload:
            try:
                payload.update(json.loads(override_payload))
            except json.JSONDecodeError:
                logger.error(f"Failed to decode {override_payload=}")
        tasks.append(asyncio.create_task(run_request(payload)))
    return await asyncio.gather(*tasks)


async def run_benchmark(args: Namespace) -> None:
    validate_args(args)
    requests = load_requests(args)
    request_url = api_url(args.base_url, "/v1/chat/completions")

    if args.debug:
        args.num_requests = 10
        args.num_warmup_requests = 3
        logger.info(f"Debug mode: only use {args.num_requests} benchmark requests")
        logger.info(f"Debug mode: only use {args.num_warmup_requests} warmup requests")

    # prune
    if args.num_requests:
        requests = requests[: args.num_requests]
        logger.info(f"Pruned to {len(requests)} requests")

    semaphore = asyncio.Semaphore(args.max_concurrency)

    async with _create_bench_client_session(
        args.max_concurrency, args.api_key
    ) as session:
        warmup_requests = requests[: args.num_warmup_requests]
        if warmup_requests:
            logger.info(f"Warming up {len(warmup_requests)} requests")
            with tqdm(total=len(warmup_requests), desc="Warmup") as progress:
                await run_requests(
                    session,
                    request_url,
                    warmup_requests,
                    args.model,
                    args.override_payload,
                    semaphore,
                    progress,
                )
            logger.info("Warming up done")

        formal_requests = requests[args.num_warmup_requests :]
        with tqdm(total=len(formal_requests), desc="Formally Running") as progress:
            benchmark_start_time = time.perf_counter()
            outputs: List[OutputMetric] = await run_requests(
                session,
                request_url,
                formal_requests,
                args.model,
                args.override_payload,
                semaphore,
                progress,
                args.request_rate,
                benchmark_start_time,
            )
            duration_s = time.perf_counter() - benchmark_start_time

    handle_outputs(
        outputs=outputs,
        duration_s=duration_s,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        completion_tokens_output_path=args.completion_tokens_output_path,
        finish_reason_length_output_path=args.finish_reason_length_output_path,
        dump_path=args.dump_path,
        metrics_path=args.metrics_path,
        label=args.label,
    )


def parse_args(args: Optional[Sequence[str]] = None) -> Namespace:
    parser = ArgumentParser(description="Benchmark router")

    parser.add_argument(
        "--num-requests",
        default=None,
        type=int,
        help="The number of requests to benchmark",
    )
    parser.add_argument(
        "--num-warmup-requests",
        default=10,
        type=int,
        help="The number of requests to warmup",
    )
    parser.add_argument(
        "--filter-constrained-grammar-requests",
        action="store_true",
        help="Filter constrained grammar requests",
    )

    parser.add_argument("--with-ts", action="store_true")
    add_common_args(parser)

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    logger.info(f"{args=}")
    set_seed(args.seed)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
