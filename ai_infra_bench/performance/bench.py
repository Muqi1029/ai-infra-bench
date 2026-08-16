# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
import asyncio
import json
import logging
import time
from argparse import Namespace
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tqdm import tqdm

from ai_infra_bench.performance.bench_utils import (
    get_request,
    handle_outputs,
    parse_args,
    set_seed,
    validate_args,
)
from ai_infra_bench.performance.core import flush_cache, request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.io import _read_json, _read_jsonl
from ai_infra_bench.utils.req import api_url, prepare_payload, tool_filter_request

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


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


async def run_requests(
    session,
    request_url: str,
    requests: Iterable[Mapping[str, Any]],
    model: str,
    override_payload: str,
    semaphore: asyncio.Semaphore | None = None,
    pbar: tqdm | None = None,
    request_rate: float = float("inf"),
    benchmark_start_time: float | None = None,
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
        if pbar is not None:
            pbar.update(1)
            if benchmark_start_time is not None and output.success:
                completion_tokens_sum += output.completion_tokens
                elapsed_s = max(time.perf_counter() - benchmark_start_time, 1e-9)
                pbar.set_postfix(
                    {"TPS": (f"{completion_tokens_sum / elapsed_s:.2f} tokens/s")}
                )
        return output

    async for payload in get_request(requests, request_rate):
        payload = prepare_payload(payload, model, override_payload)
        tasks.append(asyncio.create_task(run_request(payload)))
    return await asyncio.gather(*tasks)


async def run_benchmark(args: Namespace) -> None:
    requests = load_requests(args)
    request_url = api_url(args.base_url, "/v1/chat/completions")
    flush_cache_endpoint = f"{args.base_url}/flush_cache"

    if args.debug:
        args.num_requests = 10
        args.num_warmup_requests = 3
        logger.info(f"Debug mode: only use {args.num_requests} benchmark requests")
        logger.info(f"Debug mode: only use {args.num_warmup_requests} warmup requests")

    # prune
    if args.num_requests:
        requests = requests[: args.num_requests]
        logger.info(f"Pruned to {len(requests)} requests")

    for max_concurrency in args.max_concurrency:
        for _ in range(args.repeat):
            semaphore = asyncio.Semaphore(max_concurrency)
            async with _create_bench_client_session(
                max_concurrency, args.api_key
            ) as session:

                # warmup first
                warmup_requests = requests[: args.num_warmup_requests]
                if warmup_requests:
                    logger.info(f"Warming up {len(warmup_requests)} requests")
                    with tqdm(total=len(warmup_requests), desc="Warmup") as pbar:
                        await run_requests(
                            session,
                            request_url,
                            warmup_requests,
                            args.model,
                            args.override_payload,
                            semaphore=None,  # not set concurrency during warmup stage
                            pbar=pbar,
                        )
                    logger.info("Warming up done")

                # flush cache
                if not args.disable_flush_cache:
                    flush_cache(session, flush_cache_endpoint)
                    formal_requests = requests
                else:
                    formal_requests = requests[args.num_warmup_requests :]

                # formal run
                with tqdm(total=len(formal_requests), desc="Formally Running") as pbar:
                    benchmark_start_time = time.perf_counter()
                    outputs: List[OutputMetric] = await run_requests(
                        session,
                        request_url,
                        formal_requests,
                        args.model,
                        args.override_payload,
                        semaphore,
                        pbar,
                        args.request_rate,
                        benchmark_start_time,
                    )
                    duration_s = time.perf_counter() - benchmark_start_time

            handle_outputs(
                outputs=outputs,
                duration_s=duration_s,
                max_concurrency=max_concurrency,
                request_rate=args.request_rate,
                dump_path=args.dump_path,
                dump_content=args.dump_content,
                metrics_path=args.metrics_path,
                label=args.label,
            )


def main(args=None):
    args = parse_args(args)
    validate_args(args)
    logger.info(f"{args=}")
    set_seed(args.seed)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
