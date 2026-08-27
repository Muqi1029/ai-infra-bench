# Async benchmark runner for OpenAI-compatible endpoints.
import asyncio
import json
import logging
import random
import time
from argparse import Namespace
from datetime import datetime
from glob import glob
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tqdm import tqdm

from ai_infra_bench.performance.bench_utils import (
    compute_random_lens,
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
RANDOM_TOKEN_UPPER_BOUND = 10_000


def read_requests_with_ts(
    payload_regex_path: str, args: Namespace | None = None
) -> List[Dict]:
    timestamped_requests = []
    for file_path in sorted(glob(payload_regex_path, recursive=True)):
        logger.info(f"Reading {file_path}")
        timestamped_requests.extend(_read_json(file_path))

    timestamped_requests.sort(key=lambda item: datetime.strptime(item[0], DATE_FORMAT))
    requests = [json.loads(content) for _, content in timestamped_requests]
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
    return requests


def generate_random_requests(
    input_len: int,
    output_len: int,
    num_requests: int,
    range_ratio: float = 1.0,
) -> List[Dict]:
    input_lens = compute_random_lens(input_len, range_ratio, num_requests)
    output_lens = compute_random_lens(output_len, range_ratio, num_requests)
    return [
        {
            "prompt": random.choices(
                range(RANDOM_TOKEN_UPPER_BOUND),
                k=input_lens[i],
            ),
            "max_tokens": output_lens[i],
            "ignore_eos": True,
        }
        for i in range(num_requests)
    ]


def get_request_url(base_url: str, dataset: str | None) -> str:
    endpoint = "/v1/completions" if dataset == "random" else "/v1/chat/completions"
    return api_url(base_url, endpoint)


def read_packaged_requests(dataset: str) -> List[Dict]:
    try:
        data_package = resources.files("ai_infra_bench_dataset")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The packaged datasets are unavailable; install ai-infra-bench[data]"
        ) from error

    data_directory = data_package.joinpath("data", dataset)
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"Payload dataset {dataset!r} is not included in ai-infra-bench-dataset"
        )

    payload_resource = data_directory.joinpath("payload.jsonl")
    shard_resources = sorted(
        (
            resource
            for resource in data_directory.iterdir()
            if resource.is_file()
            and resource.name.startswith("payload-")
            and resource.name.endswith(".jsonl")
            and resource.name[len("payload-") : -len(".jsonl")].isdigit()
        ),
        key=lambda resource: resource.name,
    )
    payload_resources = shard_resources or (
        [payload_resource] if payload_resource.is_file() else []
    )
    if not payload_resources:
        raise FileNotFoundError(
            f"Payload dataset {dataset!r} is not included in ai-infra-bench-dataset"
        )

    requests = []
    for payload_resource in payload_resources:
        with resources.as_file(payload_resource) as payload_path:
            requests.extend(read_requests(str(Path(payload_path))))
    return requests


def load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "ShareGPT length settings require transformers; "
            "install ai-infra-bench[data]"
        ) from error
    return AutoTokenizer.from_pretrained(tokenizer_id)


def resize_sharegpt_requests(
    requests: List[Dict],
    input_len: int,
    output_len: int,
    tokenizer,
    num_requests: int | None = None,
) -> List[Dict]:
    requests = list(requests)
    random.shuffle(requests)

    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    target_content_len = max(1, input_len - num_special_tokens)
    resized_requests = []
    for payload in requests:
        if num_requests is not None and len(resized_requests) >= num_requests:
            break

        messages = payload.get("messages") or []
        if not messages:
            continue
        content = messages[0].get("content")
        if not isinstance(content, str):
            continue

        prompt_token_ids = tokenizer.encode(content)
        prompt_len = len(prompt_token_ids)
        if prompt_len == 0:
            continue
        if prompt_len > target_content_len:
            input_ids = prompt_token_ids[:target_content_len]
        else:
            repeat_count = (target_content_len + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * repeat_count)[:target_content_len]

        resized_payload = dict(payload)
        resized_payload["messages"] = [
            {"role": "user", "content": tokenizer.decode(input_ids)}
        ]
        resized_payload["max_tokens"] = output_len
        resized_payload["ignore_eos"] = True
        resized_requests.append(resized_payload)

    return resized_requests


def load_requests(args: Namespace) -> List[Dict]:

    if getattr(args, "payload_regex_path", None):
        if args.with_ts:
            requests = read_requests_with_ts(args.payload_regex_path)
        else:
            requests = read_requests(args.payload_regex_path)
        logger.info(f"Read {len(requests)} requests")
        if len(requests) == 0:
            logger.error(
                f"Read 0 requests! Please check your --payload-regex-path ({args.payload_regex_path=})"
            )

    elif getattr(args, "dataset", None) == "random":
        requests = generate_random_requests(
            input_len=args.input_len,
            output_len=args.output_len,
            num_requests=args.num_requests,
            range_ratio=getattr(args, "random_range_ratio", 1.0),
        )
        logger.info(f"Generated {len(requests)} random requests")
    elif getattr(args, "dataset", None) in {"gsm8k", "sharegpt"}:
        requests = read_packaged_requests(args.dataset)
        logger.info(f"Loaded {len(requests)} {args.dataset} payloads")
        if args.dataset == "sharegpt" and getattr(args, "input_len", None) is not None:
            tokenizer = load_tokenizer(
                getattr(args, "tokenizer", None) or getattr(args, "model", None)
            )
            requests = resize_sharegpt_requests(
                requests=requests,
                input_len=args.input_len,
                output_len=args.output_len,
                tokenizer=tokenizer,
                num_requests=args.num_requests,
            )
            logger.info(
                f"Prepared {len(requests)} ShareGPT requests with "
                f"input_len={args.input_len}, output_len={args.output_len}"
            )

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
            sem=semaphore,
        )
        if pbar is not None:
            pbar.update(1)
            if benchmark_start_time is not None and output.success:
                completion_tokens_sum += output.completion_tokens
                elapsed_s = max(time.perf_counter() - benchmark_start_time, 1e-9)
                average_tps = completion_tokens_sum / elapsed_s
                pbar.set_postfix({"Avg TPS": f"{average_tps:.2f} tokens/s"})
        return output

    async for payload in get_request(requests, request_rate):
        payload = prepare_payload(payload, model, override_payload)
        tasks.append(asyncio.create_task(run_request(payload)))
    return await asyncio.gather(*tasks)


async def run_benchmark(args: Namespace) -> None:
    requests = load_requests(args)
    request_url = get_request_url(
        args.base_url,
        getattr(args, "dataset", None),
    )
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
                    await flush_cache(session, flush_cache_endpoint)
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
