import asyncio
import json
import logging
import time
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import Dict, List, Optional

import aiohttp
from tqdm import tqdm

from ai_infra_bench.performance.core import request_func
from ai_infra_bench.performance.utils import (
    _create_bench_client_session,
    get_request,
    handle_outputs,
    set_seed,
    wait_for_request_interval,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def read_session_requests(payload_regex_path: str) -> List[List[Dict]]:
    session_requests = []

    num_sessions = 0
    num_requests = 0

    for filepath in sorted(glob(payload_regex_path, recursive=True)):
        logger.info(f"Reading {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            num_sessions += 1
            requests = []
            for line in f:
                if not line.strip():
                    continue
                num_requests += 1
                requests.append(json.loads(line))
            session_requests.append(requests)
    logger.info(f"Totally get {num_sessions=} {num_requests=}")
    return session_requests


async def request_func_wrapper(
    args: Namespace,
    session: aiohttp.ClientSession,
    request_url: str,
    session_payloads: List[Dict],
    sem: asyncio.Semaphore,
    pbar: Optional[tqdm] = None,
):
    session_outputs = []
    async with sem:
        # set model and stream
        # Keep payloads in the same session strictly sequential.
        for payload_index, raw_payload in enumerate(session_payloads):
            session_outputs.append(
                await request_func(args, session, request_url, raw_payload.copy())
            )

            if payload_index + 1 < len(session_payloads):
                await wait_for_request_interval(args.request_rate)

    if pbar:
        pbar.update(1)
    return session_outputs


def flatten_outputs(outputs):
    res = []
    for output in outputs:
        res.extend(output)
    return res


async def run_benchmark(args):
    # read dataset
    session_requests: List[List[Dict]] = read_session_requests(args.payload_regex_path)
    request_url = args.base_url.rstrip("/") + "/v1/chat/completions"

    if args.debug:
        args.num_sessions = 3
        logger.info(f"Debug mode: only use {args.num_sessions} session requests")

    # prune
    if args.num_sessions:
        session_requests = session_requests[: args.num_sessions]
        logger.info(f"Pruned to {args.num_sessions} sessions requests")

    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if not (args.request_rate > 0):
        raise ValueError("--request-rate must be > 0")

    sem = asyncio.Semaphore(args.max_concurrency)

    async with _create_bench_client_session(
        args.max_concurrency, args.api_key
    ) as session:
        # warmup
        if args.num_warmup_sessions:
            logger.info(f"Warming up {args.num_warmup_sessions} session requests")
            warmup_session_requests = session_requests[: args.num_warmup_sessions]
            pbar = tqdm(total=len(warmup_session_requests), desc="Warmup")
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        request_func_wrapper(
                            args, session, request_url, session_payloads, sem, pbar
                        )
                    )
                    for session_payloads in warmup_session_requests
                ]
            )
            logger.info(f"Warming up done")

        formal_run_session_requests = session_requests[args.num_warmup_sessions :]
        pbar = tqdm(
            total=len(formal_run_session_requests),
            desc="Formally running",
        )

        tasks = []
        benchmark_start_time = time.perf_counter()
        async for session_payloads in get_request(
            formal_run_session_requests, args.request_rate
        ):
            tasks.append(
                asyncio.create_task(
                    request_func(
                        args,
                        session,
                        request_url,
                        session_payloads,
                        sem,
                        pbar,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        pbar.close()
        benchmark_end_time = time.perf_counter()
        duration_s = benchmark_end_time - benchmark_start_time

    # handle outputs
    handle_outputs(
        outputs=flatten_outputs(outputs),
        duration_s=duration_s,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        completion_tokens_output_path=args.completion_tokens_output_path,
        finish_reason_length_output_path=args.finish_reason_length_output_path,
    )


def parse_args():
    parser = ArgumentParser(description="Benchmark router")
    parser.add_argument(
        "--num-sessions",
        default=None,
        type=int,
        help="The number of requests to benchmark",
    )
    parser.add_argument(
        "--num-warmup-sessions",
        default=3,
        type=int,
        help="The number of requests to warmup",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"{args=}")
    set_seed(args.seed)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
