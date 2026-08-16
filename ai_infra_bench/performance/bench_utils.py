import asyncio
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np

from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.draw import (
    format_histogram_percentages,
    format_mean,
    format_percentile,
    print_table,
)
from ai_infra_bench.utils.req import add_common_args

logger = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Benchmark")
    add_common_args(parser)

    parser.add_argument(
        "--num-requests",
        default=None,
        type=int,
        nargs="+",
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

    parser.add_argument(
        "--max-concurrency",
        default=[32],
        type=int,
        nargs="+",
        help="The max concurrency",
    )
    parser.add_argument(
        "--request-rate", default=float("inf"), type=float, help="Request rate"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat Count for each concurrency benchmark",
    )

    parser.add_argument(
        "--payload-regex-path", type=str, help="The path of requests", required=True
    )

    parser.add_argument("--label", help="Label used for discribe this benchmark")

    parser.add_argument("--with-ts", action="store_true")
    parser.add_argument("--dump-path", help="The dump path, jsonl format")
    parser.add_argument(
        "--dump-content",
        default="all",
        choices=["all", "msg"],
        help="The dump Content, jsonl format",
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        help="Optional path to dump the printed metric tables. JSON for write, JSONL for append",
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument(
        "--disable-flush-cache",
        action="store_true",
        help="Whether to disable send a flush_cache before a benchmark",
    )

    return parser.parse_args(args)


def validate_args(args: Namespace) -> None:
    for max_concurrency in args.max_concurrency:
        if max_concurrency < 1:
            raise ValueError("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be > 0")
    if args.num_warmup_requests < 0:
        raise ValueError("--num-warmup-requests must be >= 0")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")

    if (metrics_path := args.metrics_path) and not (
        any(metrics_path.endswith(suffix) for suffix in [".json", ".jsonl"])
    ):
        raise ValueError("--metrics-path must end with .json or .jsonl")

    if (dump_path := args.dump_path) and not dump_path.endswith(".jsonl"):
        logger.warning(
            "Dump path only supports jsonl format; appending the .jsonl suffix"
        )
        args.dump_path = f"{dump_path}.jsonl"


def filter_outputs(outputs: List[OutputMetric]) -> List[OutputMetric]:
    filtered_outputs = []
    for output in outputs:
        if output.success and output.prompt_tokens >= 1:
            filtered_outputs.append(output)
    return filtered_outputs


def maybe_dump_outputs(
    outputs: List[OutputMetric], dump_path: str | None, dump_content: str
) -> None:
    if not dump_path:
        return

    logger.info(f"Dumping all {len(outputs)} outputs to {dump_path}")
    with open(dump_path, "w", encoding="utf-8") as f:
        for output in outputs:
            if dump_content == "all":
                f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")
            elif dump_content == "msg":
                msg = output.payload["messages"]
                msg.append({"role": "assistant", "content": output.content})
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def maybe_dump_metric_tables(
    metric_tables: Dict[str, List[Dict[str, Any]]], metrics_path: str | None
) -> None:
    if not metrics_path:
        return
    normalized_path = metrics_path.lower()
    if normalized_path.endswith(".json"):
        logger.info(f"Writing metrics to {metrics_path}")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metric_tables, f, ensure_ascii=False, indent=2)
    elif normalized_path.endswith(".jsonl"):
        logger.info(f"Appending metrics to {metrics_path}")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_tables, ensure_ascii=False) + "\n")


def handle_outputs(
    outputs: List[OutputMetric],
    duration_s: float,
    max_concurrency: int,
    request_rate: float,
    dump_path: str | None = None,
    dump_content: str | None = None,
    metrics_path: str | None = None,
    label: str | None = None,
) -> Dict:
    maybe_dump_outputs(outputs, dump_path, dump_content)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_tables: Dict[str, str | List[Dict[str, Any]]] = {
        "label": f"{label}({now})" if label else now
    }

    def emit_metric_table(title: str, rows: List[List[Any]]) -> None:
        print_table(title, rows)
        headers = rows[0]
        metric_tables[title] = [
            {str(header): value for header, value in zip(headers, row)}
            for row in rows[1:]
        ]

    # filter failed requests
    filtered_outputs = filter_outputs(outputs)
    num_total_requests = len(outputs)
    num_success_requests = len(filtered_outputs)
    num_failed_requests = num_total_requests - num_success_requests
    if len(filtered_outputs) != len(outputs):
        if num_failed_requests > 0:
            logger.warning(f"Failed requests: {num_failed_requests}")
    if not filtered_outputs:
        emit_metric_table(
            "Benchmark Results",
            [
                ["Metric", "Value"],
                ["Total requests", str(num_total_requests)],
                ["Successful requests", "0"],
                ["Failed requests", str(num_failed_requests)],
                ["Status", "No successful requests"],
            ],
        )
        maybe_dump_metric_tables(metric_tables, metrics_path)
        return

    # latency
    ttft_ms_list = [output.ttft_ms for output in filtered_outputs]
    tpot_ms_list = [
        tpot_ms
        for output in filtered_outputs
        if (tpot_ms := output.calculate_tpot_ms()) is not None
    ]
    latency_ms_list = [output.latency_ms for output in filtered_outputs]

    # token usage
    prompt_tokens_list = [output.prompt_tokens for output in filtered_outputs]
    reasoning_tokens_list = [output.reasoning_tokens for output in filtered_outputs]
    completion_tokens_list = [output.completion_tokens for output in filtered_outputs]
    total_prompt_tokens = sum(prompt_tokens_list)
    total_reasoning_tokens = sum(reasoning_tokens_list)
    total_completion_tokens = sum(completion_tokens_list)

    # cached tokens
    cached_tokens_list = [output.cached_tokens for output in filtered_outputs]
    total_cached_tokens = sum(cached_tokens_list)
    total_cached_tokens_device = sum(
        [output.cached_tokens_device for output in filtered_outputs]
    )
    total_cached_tokens_host = sum(
        [output.cached_tokens_host for output in filtered_outputs]
    )
    cached_tokens_device_ratio = (
        total_cached_tokens_device / total_cached_tokens if total_cached_tokens else 0.0
    )
    cached_tokens_host_ratio = (
        total_cached_tokens_host / total_cached_tokens if total_cached_tokens else 0.0
    )

    # Match the per-request cache ratio denominator: prompt_tokens - 1.
    cached_tokens_ratio_list = [
        output.cached_tokens / max(output.prompt_tokens - 1, 1)
        for output in filtered_outputs
    ]

    total_cacheable_prompt_tokens = total_prompt_tokens - num_success_requests
    global_cache_ratio = (
        total_cached_tokens / total_cacheable_prompt_tokens
        if total_cacheable_prompt_tokens > 0
        else 0.0
    )

    # basic info
    duration_s = max(duration_s, 1e-9)
    finished_requests_per_second = num_success_requests / duration_s
    output_throughput = total_completion_tokens / duration_s
    request_rate_display = (
        "unlimited" if request_rate == float("inf") else f"{request_rate:g} req/s"
    )

    # device_name, device_memory = get_first_gpu_info()

    emit_metric_table(
        "Benchmark Summary",
        [
            ["Metric", "Value"],
            # ["Device info", device_name],
            # ["Device memory", device_memory],
            ["Total requests", str(num_total_requests)],
            ["Successful requests", str(num_success_requests)],
            ["Failed requests", str(num_failed_requests)],
            ["Max concurrency", str(max_concurrency)],
            ["Request rate", request_rate_display],
            ["Duration", f"{duration_s:.2f} s"],
            [
                "Mean finished requests per second",
                f"{finished_requests_per_second:.2f} req/s",
            ],
            ["Output throughput", f"{output_throughput:.2f} tokens/s"],
            ["Total prompt tokens", f"{total_prompt_tokens} tokens"],
            ["Total completion tokens", f"{total_completion_tokens} tokens"],
            ["Total reasoning tokens", f"{total_reasoning_tokens} tokens"],
            ["Total cached tokens", f"{total_cached_tokens} tokens"],
            [
                "Total cached tokens device",
                f"{total_cached_tokens_device} ({cached_tokens_device_ratio:.2%}) tokens",
            ],
            [
                "Total cached tokens host",
                f"{total_cached_tokens_host} ({cached_tokens_host_ratio:.2%}) tokens",
            ],
            ["Global cache ratio", f"{global_cache_ratio:.2%}"],
        ],
    )

    def compute_metrics(numeric_metrics: List):
        return [
            format_mean(numeric_metrics),
            format_percentile(numeric_metrics, 50),
            format_percentile(numeric_metrics, 95),
            format_percentile(numeric_metrics, 99),
        ]

    # latency
    emit_metric_table(
        "Latency & Token Metrics",
        [
            ["Metric", "Mean", "P50", "P95", "P99", "Unit"],
            [
                "TTFT",
                *compute_metrics(ttft_ms_list),
                "ms",
            ],
            [
                "TPOT(ecl the ttft)",
                *compute_metrics(tpot_ms_list),
                "ms",
            ],
            [
                "Latency",
                *compute_metrics(latency_ms_list),
                "ms",
            ],
            [
                "Prompt tokens",
                *compute_metrics(prompt_tokens_list),
                "tokens",
            ],
            ["Reasoning tokens", *compute_metrics(reasoning_tokens_list), "tokens"],
            [
                "Cached tokens",
                *compute_metrics(cached_tokens_list),
                "tokens",
            ],
            [
                "Completion tokens",
                *compute_metrics(completion_tokens_list),
                "tokens",
            ],
            [
                "Cached token ratio",
                f"{np.mean(cached_tokens_ratio_list):.2%}",
                f"{np.percentile(cached_tokens_ratio_list, 50):.2%}",
                f"{np.percentile(cached_tokens_ratio_list, 95):.2%}",
                f"{np.percentile(cached_tokens_ratio_list, 99):.2%}",
                "ratio",
            ],
        ],
    )

    # spec tokens
    total_spec_num_proposed_drafts = sum(
        [output.spec_num_proposed_drafts for output in filtered_outputs]
    )
    if total_spec_num_proposed_drafts != 0:
        # compute avg spec accept rate
        total_spec_num_correct_drafts = sum(
            [output.spec_num_correct_drafts for output in filtered_outputs]
        )
        avg_spec_accept_rate = (
            total_spec_num_correct_drafts / total_spec_num_proposed_drafts
        )

        # compute avg spec accept length
        total_spec_verify_ct = sum(output.spec_verify_ct for output in filtered_outputs)
        total_spec_completion_tokens = sum(
            output.completion_tokens for output in filtered_outputs
        )
        avg_spec_accept_length = (
            f"{total_spec_completion_tokens / total_spec_verify_ct:.2f}"
            if total_spec_verify_ct
            else "N/A"
        )

        # compute histogram
        max_length_hist = max(
            [len(output.spec_correct_drafts_histogram) for output in filtered_outputs]
        )
        spec_correct_drafts_histogram_list = [
            output.spec_correct_drafts_histogram
            + [0] * (max_length_hist - len(output.spec_correct_drafts_histogram))
            for output in filtered_outputs
        ]
        spec_correct_drafts_histogram_arr = np.array(spec_correct_drafts_histogram_list)
        total_spec_correct_drafts_histogram = np.sum(
            spec_correct_drafts_histogram_arr, axis=0
        ).tolist()
        emit_metric_table(
            "Spec Tokens Statistics",
            [
                ["Metric", "Value"],
                ["Avg Spec Accept Rate", f"{avg_spec_accept_rate:.2%}"],
                ["Avg Spec Accept Length", avg_spec_accept_length],
                [
                    "Total Spec Correct Drafts Histogram",
                    total_spec_correct_drafts_histogram,
                ],
                [
                    "Spec Correct Drafts Histogram Percentages",
                    format_histogram_percentages(total_spec_correct_drafts_histogram),
                ],
            ],
        )

    # finish reason
    finish_reasons = ("stop", "length", "tool_calls", "abort")
    finish_reason_counts = {
        finish_reason: sum(
            output.finish_reason == finish_reason for output in filtered_outputs
        )
        for finish_reason in finish_reasons
    }
    emit_metric_table(
        "Finish Reason Statistics",
        [
            ["Finish reason", "Requests", "Percentage"],
            *[
                [
                    finish_reason,
                    str(count),
                    f"{count / num_success_requests:.2%}",
                ]
                for finish_reason, count in finish_reason_counts.items()
            ],
        ],
    )

    maybe_dump_metric_tables(metric_tables, metrics_path)

    return metric_tables


async def wait_for_request_interval(request_rate: float) -> None:
    if request_rate == float("inf"):
        return

    interval = np.random.exponential(1.0 / request_rate)
    await asyncio.sleep(interval)


async def get_request(requests, request_rate):
    iterator = iter(requests)
    try:
        yield next(iterator)
    except StopIteration:
        return

    for req in iterator:
        await wait_for_request_interval(request_rate)
        yield req


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
