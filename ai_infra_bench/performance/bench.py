# Async benchmark runner for OpenAI-compatible endpoints.
import asyncio
import json
import logging
import random
import time
import uuid
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping

from ai_infra_bench.performance.bench_utils import (
    compute_random_lens,
    compute_shared_prefix_len,
    get_request,
    handle_outputs,
    parse_args,
    parse_goodput,
    prepare_warmup_requests,
    set_seed,
    validate_args,
)
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.io import (
    DATE_FORMAT,
    _read_json,
    read_packaged_requests,
    read_requests,
    read_requests_with_ts,
)
from ai_infra_bench.utils.req import api_url, prepare_payload, tool_filter_request

if TYPE_CHECKING:
    from tqdm import tqdm

logger = logging.getLogger(__name__)

RANDOM_TOKEN_UPPER_BOUND = 10_000


def _create_bench_client_session(*args, **kwargs):
    from ai_infra_bench.utils.client import (
        _create_bench_client_session as _create_session,
    )

    return _create_session(*args, **kwargs)


async def request_func(*args, **kwargs):
    from ai_infra_bench.performance.core import request_func as _request_func

    return await _request_func(*args, **kwargs)


async def flush_cache(*args, **kwargs):
    from ai_infra_bench.performance.core import flush_cache as _flush_cache

    return await _flush_cache(*args, **kwargs)


def generate_random_requests(
    input_len: int,
    output_len: int,
    num_requests: int,
    range_ratio: float = 1.0,
    cache_ratio: float = 0.0,
) -> List[Dict]:
    input_lens = compute_random_lens(input_len, range_ratio, num_requests)
    output_lens = compute_random_lens(output_len, range_ratio, num_requests)
    prefix_len = compute_shared_prefix_len(input_len, cache_ratio)
    if prefix_len and input_lens and prefix_len >= min(input_lens):
        raise ValueError(
            "shared prefix length exceeds a sampled input's cacheable length; "
            "lower --cache-ratio or increase --random-range-ratio"
        )
    prefix = random.choices(range(RANDOM_TOKEN_UPPER_BOUND), k=prefix_len)
    return [
        {
            "prompt": prefix
            + random.choices(
                range(RANDOM_TOKEN_UPPER_BOUND), k=input_lens[i] - prefix_len
            ),
            "max_tokens": output_lens[i],
            "ignore_eos": True,
        }
        for i in range(num_requests)
    ]


def get_request_url(base_url: str, dataset: str | None) -> str:
    endpoint = "/v1/completions" if dataset == "random" else "/v1/chat/completions"
    return api_url(base_url, endpoint)


def request_url_for_payloads(base_url: str, requests: List[Mapping[str, Any]]) -> str:
    """Choose the OpenAI endpoint from the loaded payload shape."""
    if requests and "prompt" in requests[0] and "messages" not in requests[0]:
        return api_url(base_url, "/v1/completions")
    return api_url(base_url, "/v1/chat/completions")


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
    payload_path = getattr(args, "payload_path", None) or getattr(
        args, "payload_regex_path", None
    )
    if payload_path:
        if getattr(args, "with_ts", False):
            requests = read_requests_with_ts(payload_path)
        else:
            requests = read_requests(payload_path)
            if (
                len(requests) == 1
                and getattr(args, "num_requests", None)
                and (
                    payload_path == "-"
                    or Path(payload_path).is_file()
                    and payload_path.endswith(".json")
                )
            ):
                loaded_single = (
                    isinstance(_read_json(payload_path), Mapping)
                    if payload_path != "-"
                    else True
                )
                if loaded_single:
                    requests = [dict(requests[0]) for _ in range(args.num_requests)]
        logger.info(f"Read {len(requests)} requests")
        if len(requests) == 0:
            logger.error(
                f"Read 0 requests! Please check your --payload-path ({payload_path=})"
            )
    elif getattr(args, "prompt", None) is not None:
        requests = [
            {
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.output_len,
            }
            for _ in range(args.num_requests)
        ]
    elif getattr(args, "dataset", None) == "random":
        cache_ratio = getattr(args, "cache_ratio", 0.0)
        requests = generate_random_requests(
            input_len=args.input_len,
            output_len=args.output_len,
            num_requests=args.num_requests,
            range_ratio=getattr(args, "random_range_ratio", 1.0),
            cache_ratio=cache_ratio,
        )
        prefix_len = compute_shared_prefix_len(args.input_len, cache_ratio)
        if prefix_len:
            logger.info(
                f"Generated {len(requests)} random requests with a "
                f"{prefix_len}-token shared prefix"
            )
        else:
            logger.info(f"Generated {len(requests)} random requests")
    elif getattr(args, "dataset", None) in {"gsm8k", "gpqa", "sharegpt"}:
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
    else:
        raise ValueError("one of --dataset or --payload-path is required")

    if getattr(args, "filter_constrained_grammar_requests", False):
        filtered_requests = [
            request for request in requests if tool_filter_request(request)
        ]
        num_filtered_requests = len(requests) - len(filtered_requests)
        logger.info(f"Filter {num_filtered_requests} due to constrained decoding")
        requests = filtered_requests
    return requests


async def _run_warmup(
    session, request_url: str, warmup_requests: List, args: Namespace
) -> None:
    from tqdm import tqdm

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


async def run_requests(
    session,
    request_url: str,
    requests: Iterable[Mapping[str, Any]],
    model: str,
    override_payload: str,
    semaphore: asyncio.Semaphore | None = None,
    pbar: "tqdm | None" = None,
    request_rate: float = float("inf"),
    benchmark_start_time: float | None = None,
    max_concurrency: int | None = 32,
    *,
    burstiness: float = 1.0,
    run_stats: Dict[str, Any] | None = None,
) -> List[OutputMetric]:
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")

    outputs: List[Any] = []
    completion_tokens_sum = 0
    in_flight = 0
    peak_concurrency = 0
    rid_suffix = uuid.uuid4().hex

    async def run_request(payload: Dict[str, Any]) -> OutputMetric:
        nonlocal completion_tokens_sum, in_flight, peak_concurrency

        async with semaphore or nullcontext():
            payload = prepare_payload(payload, model, override_payload)
            if payload.get("rid") is not None:
                payload["rid"] = f"{payload['rid']}-{rid_suffix}"
            in_flight += 1
            peak_concurrency = max(peak_concurrency, in_flight)
            try:
                output = await request_func(session, request_url, payload)
            finally:
                in_flight -= 1
        if pbar is not None:
            pbar.update(1)
            if benchmark_start_time is not None and output.success:
                completion_tokens_sum += output.completion_tokens
                elapsed_s = max(time.perf_counter() - benchmark_start_time, 1e-9)
                average_tps = completion_tokens_sum / elapsed_s
                pbar.set_postfix({"Avg TPS": f"{average_tps:.2f} tokens/s"})
        return output

    def _arrival_stream():
        if burstiness == 1.0:
            return get_request(requests, request_rate)
        return get_request(requests, request_rate, burstiness)

    if max_concurrency is None:
        # Pure open-loop mode: dispatch every arrival immediately without a
        # client-side worker cap so server queuing is measured faithfully.
        spawned: List[asyncio.Task] = []

        async def run_indexed(idx: int, raw_payload: Mapping[str, Any]) -> None:
            outputs[idx] = await run_request(dict(raw_payload))

        try:
            async for payload in _arrival_stream():
                index = len(outputs)
                outputs.append(None)
                spawned.append(asyncio.create_task(run_indexed(index, payload)))
            if spawned:
                await asyncio.gather(*spawned)
        finally:
            for task in spawned:
                task.cancel()
            if spawned:
                await asyncio.gather(*spawned, return_exceptions=True)
            if run_stats is not None:
                run_stats["peak_concurrency"] = peak_concurrency
        return outputs

    # Rate-limited arrivals may queue behind busy workers. Keep their original
    # payloads untouched until a worker is ready. Saturation mode also bounds
    # the queue so a large input iterable is consumed incrementally.
    queue: asyncio.Queue = asyncio.Queue(
        maxsize=max_concurrency if request_rate == float("inf") else 0
    )

    async def worker():
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                index, payload = item
                outputs[index] = await run_request(payload)
            finally:
                queue.task_done()

    async def produce():
        async for payload in _arrival_stream():
            index = len(outputs)
            outputs.append(None)
            await queue.put((index, payload))
        for _ in range(max_concurrency):
            await queue.put(None)

    tasks = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
    tasks.append(asyncio.create_task(produce()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if run_stats is not None:
            run_stats["peak_concurrency"] = peak_concurrency
    return outputs


async def run_benchmark(args: Namespace) -> None:
    from tqdm import tqdm

    if args.debug:
        args.num_requests = 10
        args.num_warmup_requests = 3
        logger.info(f"Debug mode: only use {args.num_requests} benchmark requests")
        logger.info(f"Debug mode: only use {args.num_warmup_requests} warmup requests")

    requests = load_requests(args)
    request_url = request_url_for_payloads(args.base_url, requests)
    flush_cache_endpoint = api_url(args.base_url, "/flush_cache")

    # prune
    if args.num_requests:
        requests = requests[: args.num_requests]
        logger.info(f"Pruned to {len(requests)} requests")

    cache_ratio = getattr(args, "cache_ratio", 0.0)
    prefix_len = compute_shared_prefix_len(args.input_len, cache_ratio)

    # prepare warmup_requests & formal_requests
    warmup_requests = prepare_warmup_requests(requests, args)
    formal_requests = requests

    # Ordinary warmup runs once, ahead of the loop. Each measured round flushes
    # first, so that warmup does not stay in the KV cache.
    if warmup_requests and prefix_len == 0 and not args.disable_flush_cache:
        logger.info(f"Warming up {len(warmup_requests)} requests")
        async with _create_bench_client_session(
            args.max_concurrency[0], args.api_key, args.request_timeout
        ) as session:
            await _run_warmup(session, request_url, warmup_requests, args)

    goodput_thresholds = parse_goodput(getattr(args, "goodput", None))
    qps_values = args.qps if getattr(args, "qps", None) else [args.request_rate]
    burstiness = getattr(args, "burstiness", 1.0)

    for request_rate in qps_values:
        for max_concurrency in args.max_concurrency:
            for _ in range(args.repeat):
                semaphore = (
                    asyncio.Semaphore(max_concurrency)
                    if max_concurrency is not None
                    else None
                )
                run_stats: Dict[str, Any] = {}
                extra_kwargs: Dict[str, Any] = {}
                if burstiness != 1.0:
                    extra_kwargs["burstiness"] = burstiness
                if max_concurrency is None:
                    extra_kwargs["run_stats"] = run_stats

                async with _create_bench_client_session(
                    max_concurrency, args.api_key, args.request_timeout
                ) as session:
                    if not args.disable_flush_cache:
                        await flush_cache(session, flush_cache_endpoint)
                        if prefix_len > 0 and warmup_requests:
                            logger.info(
                                f"Warming up shared prefix of {prefix_len} tokens"
                            )
                            await _run_warmup(
                                session, request_url, warmup_requests, args
                            )

                    # formal run
                    with tqdm(
                        total=len(formal_requests), desc="Formally Running"
                    ) as pbar:
                        benchmark_start_time = time.perf_counter()
                        outputs: List[OutputMetric] = await run_requests(
                            session,
                            request_url,
                            formal_requests,
                            args.model,
                            args.override_payload,
                            semaphore,
                            pbar,
                            request_rate,
                            benchmark_start_time,
                            max_concurrency=max_concurrency,
                            **extra_kwargs,
                        )
                        duration_s = time.perf_counter() - benchmark_start_time

                handle_query_kwargs: Dict[str, Any] = {}
                if "peak_concurrency" in run_stats:
                    handle_query_kwargs["peak_concurrency"] = run_stats[
                        "peak_concurrency"
                    ]
                handle_outputs(
                    outputs=outputs,
                    duration_s=duration_s,
                    max_concurrency=max_concurrency,
                    request_rate=request_rate,
                    dump_path=args.dump_path,
                    dump_content=args.dump_content,
                    dump_finish_reason_length=args.dump_finish_reason_length,
                    metric_path=args.metric_path,
                    label=args.label,
                    goodput_thresholds=goodput_thresholds,
                    **handle_query_kwargs,
                )


def main(args=None):
    args = parse_args(args)
    validate_args(args)
    logger.info(f"{args=}")
    set_seed(args.seed)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
