import asyncio
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
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
from ai_infra_bench.utils.req import add_common_args, parse_override_payload

logger = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(prog="aib bench", description="Benchmark")
    add_common_args(parser)

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

    # dataset
    mutex_data_group = parser.add_mutually_exclusive_group()
    mutex_data_group.add_argument(
        "--dataset",
        choices=["random", "gsm8k", "sharegpt"],
        help="use the dataset to benchmark",
    )
    # for random dataset
    parser.add_argument(
        "--input-len",
        type=int,
        help="Target input token length for random or ShareGPT datasets",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        help="Target output token length for random or ShareGPT datasets",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer name or path for ShareGPT; defaults to --model",
    )
    mutex_data_group.add_argument(
        "--payload-regex-path",
        type=str,
        help="The path of payloads requests",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat Count for each concurrency benchmark",
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
    if override_payload := getattr(args, "override_payload", None):
        parse_override_payload(override_payload)

    max_concurrencies = (
        args.max_concurrency
        if isinstance(args.max_concurrency, (list, tuple))
        else [args.max_concurrency]
    )
    for max_concurrency in max_concurrencies:
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

    if getattr(args, "dataset", None) == "random":
        if (
            args.input_len is None
            or args.output_len is None
            or args.num_requests is None
        ):
            raise ValueError(
                "--input-len, --output-len, --num-requests must be provided if using random dataset"
            )
        if args.input_len < 1:
            raise ValueError("--input-len must be >= 1")
        if args.output_len < 1:
            raise ValueError("--output-len must be >= 1")

    if getattr(args, "dataset", None) == "sharegpt" and (
        args.input_len is not None or args.output_len is not None
    ):
        if args.input_len is None or args.output_len is None:
            raise ValueError(
                "--input-len and --output-len must be provided together for ShareGPT"
            )
        if args.input_len < 1:
            raise ValueError("--input-len must be >= 1")
        if args.output_len < 1:
            raise ValueError("--output-len must be >= 1")
        if not getattr(args, "tokenizer", None) and not getattr(args, "model", None):
            raise ValueError(
                "--tokenizer or --model must be provided when setting ShareGPT lengths"
            )


def maybe_dump_outputs(
    outputs: List[OutputMetric], dump_path: str | None, dump_content: str
) -> None:
    if not dump_path:
        return
    if not dump_path.lower().endswith(".jsonl"):
        dump_path = f"{dump_path}.jsonl"
    dump_content = dump_content or "all"
    if dump_content not in {"all", "msg"}:
        raise ValueError("--dump-content must be all or msg")

    logger.info(f"Dumping all {len(outputs)} outputs to {dump_path}")
    with open(dump_path, "w", encoding="utf-8") as f:
        for output in outputs:
            if dump_content == "all":
                f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")
            elif dump_content == "msg":
                messages = list(output.payload["messages"])
                messages.append({"role": "assistant", "content": output.content})
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")


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
    else:
        raise ValueError("--metrics-path must end with .json or .jsonl")


MetricRows = List[List[Any]]
MetricTable = tuple[str, MetricRows]


@dataclass
class _OutputStats:
    all_outputs: List[OutputMetric]
    duration_s: float
    max_concurrency: int
    request_rate: float
    outputs: List[OutputMetric] = field(init=False)

    def __post_init__(self) -> None:
        self.outputs = [output for output in self.all_outputs if output.success]
        self.duration_s = max(self.duration_s, 1e-9)

    @property
    def num_total_requests(self) -> int:
        return len(self.all_outputs)

    @property
    def num_success_requests(self) -> int:
        return len(self.outputs)

    @property
    def num_failed_requests(self) -> int:
        return self.num_total_requests - self.num_success_requests

    @property
    def cached_token_ratios(self) -> List[float]:
        return [
            output.cached_tokens / max(output.prompt_tokens - 1, 1)
            for output in self.outputs
        ]

    def metric_series(self) -> List[tuple[str, List[float | int], str]]:
        return [
            (
                "TTFT",
                [output.ttft_ms for output in self.outputs if output.ttft_ms],
                "ms",
            ),
            (
                "TPOT(ecl the ttft)",
                [
                    tpot_ms
                    for output in self.outputs
                    if (tpot_ms := output.calculate_tpot_ms()) is not None
                ],
                "ms",
            ),
            ("Latency", [output.latency_ms for output in self.outputs], "ms"),
            (
                "Prompt tokens",
                [output.prompt_tokens for output in self.outputs],
                "tokens",
            ),
            (
                "Reasoning tokens",
                [output.reasoning_tokens for output in self.outputs],
                "tokens",
            ),
            (
                "Cached tokens",
                [output.cached_tokens for output in self.outputs],
                "tokens",
            ),
            (
                "Completion tokens",
                [output.completion_tokens for output in self.outputs],
                "tokens",
            ),
        ]


def _build_spec_table(outputs: List[OutputMetric]) -> MetricTable | None:
    total_proposed_drafts = sum(output.spec_num_proposed_drafts for output in outputs)
    if not total_proposed_drafts:
        return None

    total_correct_drafts = sum(output.spec_num_correct_drafts for output in outputs)
    spec_outputs = [output for output in outputs if output.spec_verify_ct]
    total_verify_ct = sum(output.spec_verify_ct for output in spec_outputs)
    total_completion_tokens = sum(output.completion_tokens for output in spec_outputs)
    avg_accept_length = (
        f"{total_completion_tokens / total_verify_ct:.2f}" if total_verify_ct else "N/A"
    )

    max_histogram_length = max(
        (len(output.spec_correct_drafts_histogram) for output in outputs),
        default=0,
    )
    total_histogram = [
        sum(
            output.spec_correct_drafts_histogram[index]
            for output in outputs
            if index < len(output.spec_correct_drafts_histogram)
        )
        for index in range(max_histogram_length)
    ]
    return (
        "Spec Tokens Statistics",
        [
            ["Metric", "Value"],
            [
                "Avg Spec Accept Rate",
                f"{total_correct_drafts / total_proposed_drafts:.2%}",
            ],
            ["Avg Spec Accept Length", avg_accept_length],
            ["Total Spec Correct Drafts Histogram", total_histogram],
            [
                "Spec Correct Drafts Histogram Percentages",
                format_histogram_percentages(total_histogram),
            ],
        ],
    )


def _build_request_tables(stats: _OutputStats) -> List[MetricTable]:
    if not stats.outputs:
        error_message = (
            stats.all_outputs[0].error_message if stats.all_outputs else None
        )
        return [
            (
                "Request Result",
                [
                    ["Metric", "Value"],
                    ["Status", "Failed"],
                    ["Error", error_message or "Unknown error"],
                ],
            )
        ]

    total_completion_tokens = sum(output.completion_tokens for output in stats.outputs)
    tables = [
        (
            "Request Result",
            [
                ["Metric", "Value"],
                ["Status", "Success"],
                ["Finish reason", stats.outputs[0].finish_reason or "N/A"],
                [
                    "TPS",
                    f"{total_completion_tokens / stats.duration_s:.2f} tokens/s",
                ],
            ],
        ),
        (
            "Latency & Token Metrics",
            [
                ["Metric", "Value", "Unit"],
                *[
                    [metric, format_mean(values), unit]
                    for metric, values, unit in stats.metric_series()
                ],
                [
                    "Cached token ratio",
                    f"{np.mean(stats.cached_token_ratios):.2%}",
                    "ratio",
                ],
            ],
        ),
    ]
    if spec_table := _build_spec_table(stats.outputs):
        tables.append(spec_table)
    return tables


def _build_benchmark_tables(stats: _OutputStats) -> List[MetricTable]:
    if not stats.outputs:
        return [
            (
                "Benchmark Results",
                [
                    ["Metric", "Value"],
                    ["Total requests", str(stats.num_total_requests)],
                    ["Successful requests", "0"],
                    ["Failed requests", str(stats.num_failed_requests)],
                    ["Status", "No successful requests"],
                ],
            )
        ]

    total_prompt_tokens = sum(output.prompt_tokens for output in stats.outputs)
    total_completion_tokens = sum(output.completion_tokens for output in stats.outputs)
    total_reasoning_tokens = sum(output.reasoning_tokens for output in stats.outputs)
    total_cached_tokens = sum(output.cached_tokens for output in stats.outputs)
    total_cached_tokens_device = sum(
        output.cached_tokens_device for output in stats.outputs
    )
    total_cached_tokens_host = sum(
        output.cached_tokens_host for output in stats.outputs
    )
    cached_tokens_device_ratio = (
        total_cached_tokens_device / total_cached_tokens if total_cached_tokens else 0.0
    )
    cached_tokens_host_ratio = (
        total_cached_tokens_host / total_cached_tokens if total_cached_tokens else 0.0
    )
    total_cacheable_prompt_tokens = total_prompt_tokens - stats.num_success_requests
    global_cache_ratio = (
        total_cached_tokens / total_cacheable_prompt_tokens
        if total_cacheable_prompt_tokens > 0
        else 0.0
    )
    request_rate_display = (
        "unlimited"
        if stats.request_rate == float("inf")
        else f"{stats.request_rate:g} req/s"
    )

    def compute_metrics(values: List[float | int]) -> List[str]:
        return [
            format_mean(values),
            format_percentile(values, 50),
            format_percentile(values, 95),
            format_percentile(values, 99),
        ]

    finish_reasons = ("stop", "length", "tool_calls", "abort")
    finish_reason_counts = {
        finish_reason: sum(
            output.finish_reason == finish_reason for output in stats.outputs
        )
        for finish_reason in finish_reasons
    }
    tables = [
        (
            "Benchmark Summary",
            [
                ["Metric", "Value"],
                ["Total requests", str(stats.num_total_requests)],
                ["Successful requests", str(stats.num_success_requests)],
                ["Failed requests", str(stats.num_failed_requests)],
                ["Max concurrency", str(stats.max_concurrency)],
                ["Request rate", request_rate_display],
                [
                    "Mean finished requests per second",
                    f"{stats.num_success_requests / stats.duration_s:.2f} req/s",
                ],
                ["Duration", f"{stats.duration_s:.2f} s"],
                [
                    "Output throughput",
                    f"{total_completion_tokens / stats.duration_s:.2f} tokens/s",
                ],
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
        ),
        (
            "Latency & Token Metrics",
            [
                ["Metric", "Mean", "P50", "P95", "P99", "Unit"],
                *[
                    [metric, *compute_metrics(values), unit]
                    for metric, values, unit in stats.metric_series()
                ],
                [
                    "Cached token ratio",
                    f"{np.mean(stats.cached_token_ratios):.2%}",
                    f"{np.percentile(stats.cached_token_ratios, 50):.2%}",
                    f"{np.percentile(stats.cached_token_ratios, 95):.2%}",
                    f"{np.percentile(stats.cached_token_ratios, 99):.2%}",
                    "ratio",
                ],
            ],
        ),
    ]
    if spec_table := _build_spec_table(stats.outputs):
        tables.append(spec_table)
    tables.append(
        (
            "Finish Reason Statistics",
            [
                ["Finish reason", "Requests", "Percentage"],
                *[
                    [
                        finish_reason,
                        str(count),
                        f"{count / stats.num_success_requests:.2%}",
                    ]
                    for finish_reason, count in finish_reason_counts.items()
                ],
            ],
        )
    )
    return tables


def handle_outputs(
    outputs: List[OutputMetric],
    duration_s: float,
    max_concurrency: int,
    request_rate: float,
    dump_path: str | None = None,
    dump_content: str = "all",
    metrics_path: str | None = None,
    label: str | None = None,
    benchmark_mode: bool = True,
) -> Dict:
    maybe_dump_outputs(outputs, dump_path, dump_content)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_tables: Dict[str, Any] = {
        "label": label or "benchmark",
        "timestamp": now,
        "max_concurrency": max_concurrency,
    }

    def emit_metric_table(title: str, rows: List[List[Any]]) -> None:
        print_table(title, rows)
        headers = rows[0]
        metric_tables[title] = [
            {str(header): value for header, value in zip(headers, row)}
            for row in rows[1:]
        ]

    stats = _OutputStats(outputs, duration_s, max_concurrency, request_rate)
    if stats.num_failed_requests:
        logger.warning(f"Failed requests: {stats.num_failed_requests}")

    table_builder = _build_benchmark_tables if benchmark_mode else _build_request_tables
    for title, rows in table_builder(stats):
        emit_metric_table(title, rows)

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
