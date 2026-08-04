# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py
import asyncio
import json
import logging
import time
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from typing import Dict, List

from tqdm import tqdm

from ai_infra_bench.performance.bench_utils import get_request, handle_outputs, set_seed
from ai_infra_bench.performance.common_args import add_common_args
from ai_infra_bench.performance.core import request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import normalize_payload

logger = logging.getLogger(__name__)


DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def tool_filter_request(req: dict):
    # the most strict grammar
    if req["tool_choices"] == "required" or isinstance(req["tool_choices"], dict):
        return False
    if any([tool["strict"] for tool in req["tools"]]):
        return False
    if req["response_format"]:
        return False
    return True


def read_requests_with_ts(payload_regex_path: str, args) -> List[Dict]:
    data: Dict[str, str] = {}
    file_paths = sorted(glob(payload_regex_path))
    for file_path in file_paths:
        logger.info(f"Reading {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                ts, content = item
                data[ts] = content
    sorted_items = sorted(
        data.items(), key=lambda x: datetime.strptime(x[0], DATE_FORMAT)
    )
    requests = [json.loads(content) for _, content in sorted_items]
    if args.filter_constrained_grammar_requests:
        filtered_requests = [req for req in requests if tool_filter_request(req)]
        num_filtered_requests = len(requests) - len(filtered_requests)
        logger.info(f"Filter {num_filtered_requests} due to constrained decoding")
        requests = filtered_requests
    logger.info(f"Read {len(requests)} requests")
    return requests


def read_requests(payload_regex_path: str) -> List[Dict]:
    requests = []
    for file_path in glob(payload_regex_path, recursive=True):
        with open(file_path, "r", encoding="utf-8") as f:
            reqs = json.load(f)
            requests.extend(reqs)
    return requests


async def run_benchmark(args):
    # read dataset
    if args.with_ts:
        requests = read_requests_with_ts(args.payload_regex_path, args)
    else:
        requests = read_requests(args.payload_regex_path, args)
    request_url = args.base_url + "/v1/chat/completions"

    if args.debug:
        args.num_requests = 10
        args.num_warmup_requests = 3
        logger.info(f"Debug mode: only use {args.num_requests} benchmark requests")
        logger.info(f"Debug mode: only use {args.num_warmup_requests} warmup requests")

    # prune
    if args.num_requests:
        requests = requests[: args.num_requests]
        logger.info(f"Pruned to {len(requests)} requests")

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be > 0")

    sem = asyncio.Semaphore(args.max_concurrency)

    async with _create_bench_client_session(
        args.max_concurrency, args.api_key
    ) as session:
        # warmup
        if args.num_warmup_requests:
            pbar = tqdm(total=args.num_warmup_requests, desc="Warmup")
            logger.info(f"Warming up {args.num_warmup_requests} requests")
            warmup_requests = requests[: args.num_warmup_requests]
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        request_func(
                            args,
                            session,
                            request_url,
                            normalize_payload(payload),
                            sem,
                            pbar,
                        )
                    )
                    for payload in warmup_requests
                ]
            )
            logger.info(f"Warming up done")

        formal_run_requests = requests[args.num_warmup_requests :]
        pbar = tqdm(total=len(formal_run_requests), desc="Formally Running")
        tasks = []
        benchmark_start_time = time.perf_counter()
        async for payload in get_request(formal_run_requests, args.request_rate):
            payload["model"] = args.model
            tasks.append(
                asyncio.create_task(
                    request_func(
                        session,
                        request_url,
                        normalize_payload(payload),
                        sem,
                        pbar,
                    )
                )
            )
        outputs: List[OutputMetric] = await asyncio.gather(*tasks)
        benchmark_end_time = time.perf_counter()
        duration_s = benchmark_end_time - benchmark_start_time

    # handle outputs
    handle_outputs(
        outputs=outputs,
        duration_s=duration_s,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        completion_tokens_output_path=args.completion_tokens_output_path,
        finish_reason_length_output_path=args.finish_reason_length_output_path,
    )


def parse_args(args=None):
    parser = ArgumentParser(description="Benchmark router")

    parser.add_argument(
        "--num-requests",
        default=None,
        type=int,
        help="The number of requests to benchmark",
    )
    parser.add_argument(
        "--num-warmup-requests",
        default=100,
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
