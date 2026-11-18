import asyncio
import json
import logging
import math
import os
import random
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence

from ai_infra_bench.performance.struct import MetricTable, OutputMetric, _OutputStats
from ai_infra_bench.utils.req import add_common_args, parse_override_payload
from ai_infra_bench.utils.table import format_mean, mean_value, print_table

logger = logging.getLogger(__name__)

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60
GOODPUT_METRICS = {"ttft", "tpot", "itl", "e2el", "latency"}


def parse_goodput(items: Sequence[str] | None) -> Dict[str, float]:
    if not items:
        return {}
    thresholds: Dict[str, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--goodput must use METRIC:MS format, got {item!r}")
        metric, raw_value = item.split(":", 1)
        key = metric.strip().lower()
        if key not in GOODPUT_METRICS:
            raise ValueError(
                f"--goodput metric must be one of {sorted(GOODPUT_METRICS)}, got {metric!r}"
            )
        try:
            limit = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"--goodput threshold must be numeric: {item!r}"
            ) from error
        if limit <= 0:
            raise ValueError(f"--goodput threshold must be > 0: {item!r}")
        thresholds["e2el" if key == "latency" else key] = limit
    return thresholds


def prepare_warmup_requests(requests: List[Dict], args: Namespace) -> List[Dict]:
    """Build the warmup payloads requested by the benchmark configuration."""
    cache_ratio = getattr(args, "cache_ratio", 0.0)
    prefix_len = compute_shared_prefix_len(getattr(args, "input_len", 0), cache_ratio)
    if prefix_len and requests:
        # special case for cache-ratio warmup
        return [
            {
                "prompt": requests[0]["prompt"][:prefix_len],
                "max_tokens": 1,
                "ignore_eos": True,
            }
            for _ in range(args.num_warmup_requests)
        ]

    warmup_requests = requests[: args.num_warmup_requests]
    return warmup_requests


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
    """Sample ``num`` integer lengths in the configured target range.

    ``full_len <= 0`` is valid for workloads that do not generate tokens. A
    positive target always has a lower bound of one token, matching SGLang's
    random-range behavior.
    """
    if full_len <= 0:
        return [0] * num
    low = max(int(full_len * range_ratio), 1)
    return [random.randint(low, full_len) for _ in range(num)]


def compute_shared_prefix_len(input_len: int, cache_ratio: float) -> int:
    """Return the shared prefix length for ``--cache-ratio``.

    Prefix cache can cover at most ``prompt_len - 1`` tokens, matching
    ``Global cache ratio``. The ratio is applied to that cacheable length:
    ``int((input_len - 1) * cache_ratio)``. A zero ratio disables sharing.
    """
    return int((input_len - 1) * cache_ratio)


def add_output_args(
    parser: ArgumentParser, *, default_metric_path: str | None = None
) -> None:
    parser.add_argument("--dump-path", help="The dump path, jsonl format")
    parser.add_argument(
        "--dump-content",
        default="all",
        choices=["all", "msg"],
        help="The dump Content, jsonl format",
    )
    parser.add_argument(
        "--dump-finish-reason-length",
        action="store_true",
        help="Dump requests whose finish reason is length",
    )
    parser.add_argument(
        "--metric-path",
        type=str,
        default=default_metric_path,
        help=(
            "Optional path to save benchmark metrics "
            "(.yaml/.yml appends structured YAML; .json overwrites; .jsonl appends)"
        ),
    )
    parser.add_argument(
        "--label", help="Label used to describe this benchmark, stored in metric files"
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        metavar="METRIC:MS",
        help="Per-request SLO thresholds in ms (e.g. ttft:500 tpot:50 itl:80 e2el:3000)",
    )
    parser.add_argument(
        "--report-to",
        default=os.environ.get("AIB_REPORT_URL"),
        help="Optional AIB platform URL to upload benchmark results ($AIB_REPORT_URL)",
    )


def validate_output_args(args: Namespace) -> None:
    if (metric_path := getattr(args, "metric_path", None)) and not any(
        metric_path.lower().endswith(suffix)
        for suffix in (".yaml", ".yml", ".json", ".jsonl")
    ):
        raise ValueError("--metric-path must end with .yaml, .yml, .json or .jsonl")

    if getattr(args, "dump_finish_reason_length", False) and not args.dump_path:
        args.dump_path = "aib-dump.jsonl"
        logger.warning(
            f"--dump-finish-reason-length has been set, but --dump-path not set. To ensure the dump process work as expected, --dump-path has been set to {args.dump_path}"
        )

    if (
        dump_path := getattr(args, "dump_path", None)
    ) and not dump_path.lower().endswith(".jsonl"):
        logger.warning(
            "Dump path only supports jsonl format; appending the .jsonl suffix"
        )
        args.dump_path = f"{dump_path}.jsonl"

    parse_goodput(getattr(args, "goodput", None))


def parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(prog="aib bench", description="Benchmark")
    add_common_args(parser)
    add_output_args(parser, default_metric_path="aib-metric.yaml")

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
        help="The number of requests to warmup (0 to disable)",
    )

    parser.add_argument(
        "--filter-constrained-grammar-requests",
        action="store_true",
        help="Filter constrained grammar requests",
    )

    parser.add_argument(
        "--max-concurrency",
        default=None,
        type=int,
        nargs="+",
        help="Max concurrent requests (default: 32 in closed-loop; unlimited in --qps mode)",
    )
    parser.add_argument(
        "--qps",
        default=None,
        type=float,
        nargs="+",
        help="Open-loop target queries per second (one or more QPS rates; unlimited concurrency unless --max-concurrency is set)",
    )
    parser.add_argument(
        "--request-rate", default=float("inf"), type=float, help="Request rate (req/s)"
    )
    parser.add_argument(
        "--burstiness",
        default=1.0,
        type=float,
        help="Arrival burstiness for --qps / --request-rate: 1.0=Poisson (default), 0=constant interval (1/qps), >0=Gamma",
    )
    parser.add_argument(
        "--request-timeout",
        default=BENCH_AIOHTTP_TIMEOUT_SECONDS,
        type=float,
        help="Total timeout in seconds for each request",
    )

    # dataset
    mutex_data_group = parser.add_mutually_exclusive_group(required=False)
    mutex_data_group.add_argument(
        "--prompt",
        "--user-prompt",
        help="Repeat this user prompt using chat completions",
    )
    mutex_data_group.add_argument(
        "--payload-path",
        "--payload-regex-path",
        dest="payload_path",
        help="Path/glob to JSON or JSONL payloads, or '-' for stdin",
    )
    mutex_data_group.add_argument(
        "--dataset",
        choices=["random", "gsm8k", "gpqa", "sharegpt"],
        help="use the dataset to benchmark (default: random)",
    )

    # for random dataset
    parser.add_argument(
        "--input-len",
        type=int,
        default=8192,
        help="Target input token length for random or ShareGPT datasets",
    )
    parser.add_argument(
        "--output-len",
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum output tokens; for random/ShareGPT, the target output length",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help=(
            "Sample each random request's input/output length between this "
            "fraction of the target length and the target length"
        ),
    )
    parser.add_argument(
        "--cache-ratio",
        type=float,
        default=0.0,
        help=(
            "Share a common token prefix across random requests so later "
            "requests can hit the prefix cache. The prefix length is "
            "int((--input-len - 1) * --cache-ratio) because a request can "
            "cache at most prompt_len-1 tokens. Must be used with "
            "--dataset random. A positive value primes this prefix once "
            "before each measured run"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer name or path for ShareGPT; defaults to --model",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat Count for each concurrency benchmark",
    )

    parser.add_argument(
        "--with-ts", action="store_true", help="For payload data with timestamps"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: set num-requests to 10 and num-warmup-requests to 3",
    )

    parser.add_argument(
        "--disable-flush-cache",
        action="store_true",
        help="Whether to disable send a flush_cache before a benchmark",
    )

    parsed = parser.parse_args(args)
    if parsed.max_concurrency is None:
        parsed.max_concurrency = [None] if parsed.qps is not None else [32]
    parsed.payload_regex_path = parsed.payload_path
    return parsed


def validate_args(args: Namespace) -> None:
    if override_payload := getattr(args, "override_payload", None):
        parse_override_payload(override_payload)

    payload_path = getattr(args, "payload_path", None) or getattr(
        args, "payload_regex_path", None
    )
    if getattr(args, "prompt", None) is not None and payload_path:
        raise ValueError("--prompt cannot be combined with --payload-path")
    if payload_path:
        if getattr(args, "dataset", None):
            raise ValueError("--payload-path cannot be combined with --dataset")
        args.payload_path = payload_path
        args.payload_regex_path = payload_path

    qps = getattr(args, "qps", None)
    if qps is not None:
        qps_list = list(qps) if isinstance(qps, (list, tuple)) else [qps]
        if not qps_list or any(
            rate <= 0 or math.isinf(rate) or math.isnan(rate) for rate in qps_list
        ):
            raise ValueError("--qps values must be finite numbers > 0")
        if getattr(args, "request_rate", float("inf")) != float("inf"):
            raise ValueError("cannot specify both --qps and --request-rate")
        args.qps = [float(rate) for rate in qps_list]

    burstiness = getattr(args, "burstiness", 1.0)
    if burstiness < 0 or math.isinf(burstiness) or math.isnan(burstiness):
        raise ValueError("--burstiness must be a finite number >= 0")

    raw_mc = getattr(args, "max_concurrency", None)
    if raw_mc is None and not isinstance(raw_mc, (list, tuple)):
        args.max_concurrency = [None] if qps is not None else [32]
    max_concurrencies = (
        args.max_concurrency
        if isinstance(args.max_concurrency, (list, tuple))
        else [args.max_concurrency]
    )
    for max_concurrency in max_concurrencies:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("--max-concurrency must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be > 0")
    if getattr(args, "request_timeout", BENCH_AIOHTTP_TIMEOUT_SECONDS) <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.num_warmup_requests < 0:
        raise ValueError("--num-warmup-requests must be >= 0")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")

    validate_output_args(args)

    if getattr(args, "input_len", 1) < 1:
        raise ValueError("--input-len must be >= 1")
    if getattr(args, "output_len", 0) < 0:
        raise ValueError("--output-len must be >= 0")

    random_range_ratio = getattr(args, "random_range_ratio", 1.0)
    if not 0.0 <= random_range_ratio <= 1.0:
        raise ValueError("--random-range-ratio must be between 0 and 1")

    cache_ratio = getattr(args, "cache_ratio", 0.0)
    if not 0.0 <= cache_ratio <= 1.0:
        raise ValueError("--cache-ratio must be between 0 and 1")
    if cache_ratio:
        if getattr(args, "dataset", None) != "random":
            raise ValueError("--cache-ratio must be used with --dataset random")
        prefix_len = compute_shared_prefix_len(args.input_len, cache_ratio)
        min_cacheable_len = max(int(args.input_len * random_range_ratio), 1) - 1
        if prefix_len < 1:
            raise ValueError("--cache-ratio is too small for --input-len")
        if prefix_len > min_cacheable_len:
            raise ValueError(
                "--cache-ratio prefix length exceeds the minimum sampled input's "
                "cacheable length; lower --cache-ratio or increase "
                "--random-range-ratio"
            )

    # validate dataset
    if (
        getattr(args, "prompt", None) is None
        and not payload_path
        and not getattr(args, "dataset", None)
    ):
        args.dataset = "random"

    dataset = getattr(args, "dataset", None)
    if dataset in ["random", "sharegpt"] or getattr(args, "prompt", None) is not None:
        if args.num_requests is None:
            args.num_requests = 100
            logger.warning(
                f"--num-requests is not provided, set to {args.num_requests}"
            )

    if dataset == "sharegpt":
        if not getattr(args, "tokenizer", None) and not getattr(args, "model", None):
            raise ValueError(
                "--tokenizer or --model must be provided when setting ShareGPT lengths"
            )


def maybe_dump_outputs(
    outputs: List[OutputMetric],
    dump_path: str | None,
    dump_content: str = "all",
    dump_finish_reason_length: bool = False,
) -> None:
    if not dump_path:
        return

    additional_info = ""
    if dump_finish_reason_length:
        additional_info = "(finish_reason=length)"
        outputs = [o for o in outputs if o.finish_reason == "length"]

    if not outputs:
        logger.info("No dump data since len(outputs)=0")
        return

    logger.info(f"Dumping all {len(outputs)} outputs{additional_info} to {dump_path}")
    with open(dump_path, "w", encoding="utf-8") as f:
        for output in outputs:
            if dump_content == "all":
                f.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")
            elif dump_content == "msg":
                messages = list(output.payload.get("messages") or [])
                if not messages and "prompt" in output.payload:
                    messages = [{"role": "user", "content": output.payload["prompt"]}]
                messages.append({"role": "assistant", "content": output.content})
                f.write(json.dumps(messages, ensure_ascii=False) + "\n")


def _dump_yaml_record(record: Mapping[str, Any], metric_path: str) -> None:
    import yaml

    class _CompactDumper(yaml.SafeDumper):
        pass

    def _represent_dict(dumper: yaml.SafeDumper, data: dict) -> Any:
        keys = set(data.keys())
        is_compact_leaf = (
            bool(keys)
            and (keys <= {"mean", "p50", "p95", "p99"} or keys == {"requests", "pct"})
            and all(isinstance(v, (int, float)) for v in data.values())
        )
        return dumper.represent_mapping(
            "tag:yaml.org,2002:map", data.items(), flow_style=is_compact_leaf
        )

    def _represent_list(dumper: yaml.SafeDumper, data: list) -> Any:
        is_numeric_row = len(data) <= 16 and all(
            isinstance(x, (int, float)) for x in data
        )
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=is_numeric_row
        )

    _CompactDumper.add_representer(dict, _represent_dict)
    _CompactDumper.add_representer(list, _represent_list)

    rendered = yaml.dump(
        dict(record),
        Dumper=_CompactDumper,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )
    logger.info(f"Appending YAML metrics to {metric_path}")
    with open(metric_path, "a", encoding="utf-8") as f:
        f.write(f"---\n{rendered}")


def maybe_dump_metric_tables(
    metric_tables: Dict[str, Any],
    metric_path: str | None,
    *,
    yaml_record: Mapping[str, Any] | None = None,
) -> None:
    if not metric_path:
        return
    normalized_path = metric_path.lower()
    if normalized_path.endswith((".yaml", ".yml")):
        _dump_yaml_record(yaml_record or metric_tables, metric_path)
    elif normalized_path.endswith(".json"):
        logger.info(f"Writing metrics to {metric_path}")
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metric_tables, f, ensure_ascii=False, indent=2)
    elif normalized_path.endswith(".jsonl"):
        logger.info(f"Appending metrics to {metric_path}")
        with open(metric_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_tables, ensure_ascii=False) + "\n")
    else:
        raise ValueError("--metric-path must end with .yaml, .yml, .json or .jsonl")


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
    cached_mean = mean_value(stats.cached_token_ratios)
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
                    f"{cached_mean:.2%}" if cached_mean is not None else "N/A",
                    "ratio",
                ],
            ],
        ),
    ]
    if spec_table := stats.build_spec_table():
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

    tables = [stats.build_benchmark_summary_table(), stats.build_latency_token_table()]
    if spec_table := stats.build_spec_table():
        tables.append(spec_table)
    tables.append(stats.build_finish_reason_table())
    return tables


def handle_outputs(
    outputs: List[OutputMetric],
    duration_s: float,
    max_concurrency: int | None,
    request_rate: float,
    dump_path: str | None = None,
    dump_content: str = "all",
    dump_finish_reason_length: bool = False,
    metric_path: str | None = None,
    label: str | None = None,
    benchmark_mode: bool = True,
    goodput_thresholds: Mapping[str, float] | None = None,
    peak_concurrency: int | None = None,
) -> Dict:
    maybe_dump_outputs(outputs, dump_path, dump_content, dump_finish_reason_length)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = label or ""
    metric_tables: Dict[str, Any] = {
        "label": label,
        "timestamp": now,
        "max_concurrency": max_concurrency,
    }
    if request_rate != float("inf"):
        metric_tables["request_rate"] = request_rate
    if peak_concurrency is not None:
        metric_tables["peak_concurrency"] = peak_concurrency

    def emit_metric_table(title: str, rows: List[List[Any]]) -> None:
        print_table(title, rows)
        headers = rows[0]
        metric_tables[title] = [
            {str(header): value for header, value in zip(headers, row)}
            for row in rows[1:]
        ]

    stats = _OutputStats(
        outputs,
        duration_s,
        max_concurrency,
        request_rate,
        label=label,
        ts=now,
        goodput_thresholds=goodput_thresholds,
        peak_concurrency=peak_concurrency,
    )
    if stats.num_failed_requests:
        logger.warning(f"Failed requests: {stats.num_failed_requests}")

    table_builder = _build_benchmark_tables if benchmark_mode else _build_request_tables
    for title, rows in table_builder(stats):
        emit_metric_table(title, rows)

    yaml_record = stats.build_yaml_record() if benchmark_mode else None
    maybe_dump_metric_tables(metric_tables, metric_path, yaml_record=yaml_record)

    return metric_tables


async def wait_for_request_interval(
    request_rate: float, burstiness: float = 1.0
) -> None:
    if request_rate == float("inf"):
        return

    if burstiness == 0.0:
        interval = 1.0 / request_rate
    elif burstiness == 1.0:
        interval = random.expovariate(request_rate)
    else:
        theta = 1.0 / (request_rate * burstiness)
        interval = random.gammavariate(burstiness, theta)
    await asyncio.sleep(interval)


async def get_request(requests, request_rate, burstiness: float = 1.0):
    iterator = iter(requests)
    try:
        yield next(iterator)
    except StopIteration:
        return

    for req in iterator:
        if burstiness == 1.0:
            await wait_for_request_interval(request_rate)
        else:
            await wait_for_request_interval(request_rate, burstiness)
        yield req


def set_seed(seed: int):
    random.seed(seed)
