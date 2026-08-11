import asyncio
import json
import logging
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.device import get_first_gpu_info
from ai_infra_bench.utils.draw import print_table
from ai_infra_bench.utils.req import format_histogram_percentages

logger = logging.getLogger(__name__)


def calculate_itl_ms(output: OutputMetric) -> Optional[float]:
    if output.completion_tokens <= 1 or output.ttft_ms <= 0.0:
        return None

    generation_time_ms = output.latency_ms - output.ttft_ms
    if generation_time_ms < 0.0:
        return None
    return generation_time_ms / (output.completion_tokens - 1)


def format_mean(values: List[float], precision: int = 2) -> str:
    if not values:
        return "N/A"
    return f"{np.mean(values):.{precision}f}"


def format_percentile(
    values: List[float], percentile: float, precision: int = 2
) -> str:
    if not values:
        return "N/A"
    return f"{np.percentile(values, percentile):.{precision}f}"


def filter_outputs(outputs: List[OutputMetric]) -> List[OutputMetric]:
    filtered_outputs = []
    for output in outputs:
        if output.success and output.prompt_tokens >= 1:
            filtered_outputs.append(output)
    return filtered_outputs


def dump_outputs(outputs: List[OutputMetric], dump_path: str) -> None:
    if not dump_path.endswith(".jsonl"):
        logger.warning(
            "Dump path only supports jsonl format; appending the .jsonl suffix"
        )
        dump_path += ".jsonl"

    logger.info(f"Dumping all {len(outputs)} outputs to {dump_path}")
    with open(dump_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")


def dump_metric_tables(
    metric_tables: Dict[str, List[Dict[str, Any]]], metrics_path: str
) -> None:
    normalized_path = metrics_path.lower()
    if normalized_path.endswith(".json"):
        logger.info(f"Writing metrics to {metrics_path}")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metric_tables, f, ensure_ascii=False, indent=2)
        return
    if normalized_path.endswith(".jsonl"):
        logger.info(f"Appending metrics to {metrics_path}")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_tables, ensure_ascii=False) + "\n")
        return
    raise ValueError("--metrics-path must end with .json or .jsonl")


def handle_outputs(
    outputs: List[OutputMetric],
    duration_s: float,
    max_concurrency: int,
    request_rate: float,
    completion_tokens_output_path: Optional[str] = None,
    finish_reason_length_output_path: Optional[str] = None,
    dump_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
):
    if dump_path:
        dump_outputs(outputs, dump_path)

    metric_tables: Dict[str, List[Dict[str, Any]]] = {}

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
        if metrics_path:
            dump_metric_tables(metric_tables, metrics_path)
        return

    ttft_ms_list = [output.ttft_ms for output in filtered_outputs]
    itl_ms_list = [
        itl_ms
        for output in filtered_outputs
        if (itl_ms := calculate_itl_ms(output)) is not None
    ]
    latency_ms_list = [output.latency_ms for output in filtered_outputs]

    prompt_tokens_list = [output.prompt_tokens for output in filtered_outputs]
    total_prompt_tokens = sum(prompt_tokens_list)

    completion_tokens_list = [output.completion_tokens for output in filtered_outputs]
    reasoning_tokens_list = [output.reasoning_tokens for output in filtered_outputs]
    total_completion_tokens = sum(completion_tokens_list)
    total_reasoning_tokens = sum(reasoning_tokens_list)

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

    duration_s = max(duration_s, 1e-9)
    finished_requests_per_second = num_success_requests / duration_s
    output_throughput = total_completion_tokens / duration_s
    request_rate_display = (
        "unlimited" if request_rate == float("inf") else f"{request_rate:g} req/s"
    )
    device_name, device_memory = get_first_gpu_info()

    emit_metric_table(
        "Benchmark Summary",
        [
            ["Metric", "Value"],
            ["Device info", device_name],
            ["Device memory", device_memory],
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
                "ITL",
                *compute_metrics(itl_ms_list),
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
            [
                "Completion tokens",
                *compute_metrics(completion_tokens_list),
                "tokens",
            ],
            ["Reasoning tokens", *compute_metrics(reasoning_tokens_list), "tokens"],
            [
                "Cached tokens",
                *compute_metrics(cached_tokens_list),
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
        total_spec_num_correct_drafts = sum(
            [output.spec_num_correct_drafts for output in filtered_outputs]
        )
        total_spec_accept_rate = (
            total_spec_num_correct_drafts / total_spec_num_proposed_drafts
        )
        spec_accept_length_list = [
            output.spec_accept_length for output in filtered_outputs
        ]

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
                ["Avg Spec Accept Rate", f"{total_spec_accept_rate:.2%}"],
                ["Avg Spec Accept Length", format_mean(spec_accept_length_list)],
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

    if metrics_path:
        dump_metric_tables(metric_tables, metrics_path)

    # dump completion tokens
    if completion_tokens_output_path and completion_tokens_list:
        logger.info(
            f"Dumping {len(completion_tokens_list)} completion tokens to "
            f"{completion_tokens_output_path}"
        )
        with open(completion_tokens_output_path, mode="w", encoding="utf-8") as f:
            json.dump(completion_tokens_list, f, ensure_ascii=False, indent=2)

    # dump finish length requests
    finish_reason_length_list = [
        output for output in filtered_outputs if output.finish_reason == "length"
    ]
    if finish_reason_length_output_path and finish_reason_length_list:
        logger.info(
            f"Dumping {len(finish_reason_length_list)} finish reason 'length' to "
            f"{finish_reason_length_output_path}"
        )
        with open(finish_reason_length_output_path, mode="w", encoding="utf-8") as f:
            json.dump(
                [asdict(output) for output in finish_reason_length_list],
                f,
                ensure_ascii=False,
                indent=2,
            )


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
