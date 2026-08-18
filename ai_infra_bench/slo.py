"""YAML-driven SLO search for OpenAI-compatible endpoints."""

import argparse
import asyncio
import logging
import operator
import random
import time
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml

from ai_infra_bench.performance.bench_utils import get_request
from ai_infra_bench.performance.core import request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import api_url, prepare_payload, sanitize_url

logger = logging.getLogger(__name__)

_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}
_SERIES = (
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    # Compatibility aliases for the latency names used by the former SLO API.
    "itl_ms",
    "e2e_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
)
SUPPORTED_METRICS = {
    "total_requests",
    "successful_requests",
    "failed_requests",
    "success_rate",
    "duration_s",
    "request_throughput",
    "output_throughput",
    "max_concurrency",
    "request_rate",
    *(
        f"{prefix}_{name}"
        for prefix in ("mean", "p50", "p95", "p99")
        for name in _SERIES
    ),
}


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    api_key: str = "EMPTY"
    model: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class RequestConfig:
    payload: Dict[str, Any] | None = None
    input_len: int | None = None
    output_len: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    num_requests: int = 100
    warmup_requests: int = 0
    max_concurrency: int = 32
    request_rate: float = float("inf")
    seed: int = 42


@dataclass(frozen=True)
class SearchConfig:
    parameter: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class Condition:
    metric: str
    operator: str
    value: float


@dataclass(frozen=True)
class SLOConfig:
    endpoint: EndpointConfig
    request: RequestConfig
    benchmark: BenchmarkConfig
    search: SearchConfig
    conditions: List[Condition]


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _request_rate(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "infinity", "unlimited"}:
        return float("inf")
    if not isinstance(value, Real) or value <= 0:
        raise ValueError("benchmark.request_rate must be positive or 'inf'")
    return float(value)


def parse_slo_config(data: Mapping[str, Any]) -> SLOConfig:
    endpoint_data = _section(data, "endpoint")
    base_url = endpoint_data.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("endpoint.base_url must be a non-empty string")
    for name in ("api_key", "model", "path"):
        value = endpoint_data.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"endpoint.{name} must be a string")
    endpoint = EndpointConfig(
        base_url=sanitize_url(base_url),
        api_key=endpoint_data.get("api_key") or "EMPTY",
        model=endpoint_data.get("model"),
        path=endpoint_data.get("path"),
    )

    request_data = _section(data, "request")
    payload = request_data.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("request.payload must be a YAML mapping")
    input_len = request_data.get("input_len")
    output_len = request_data.get("output_len")
    if payload is None:
        input_len = _positive_int(input_len, "request.input_len")
        output_len = _positive_int(output_len, "request.output_len")
    request = RequestConfig(
        payload=dict(payload) if payload is not None else None,
        input_len=input_len,
        output_len=output_len,
    )

    search_data = _section(data, "search")
    parameter = search_data.get("parameter", "max_concurrency")
    if parameter not in {"max_concurrency", "request_rate"}:
        raise ValueError("search.parameter must be max_concurrency or request_rate")
    minimum = _positive_int(search_data.get("min"), "search.min")
    maximum = _positive_int(search_data.get("max"), "search.max")
    if minimum > maximum:
        raise ValueError("search.min must be <= search.max")
    search = SearchConfig(parameter, minimum, maximum)

    benchmark_data = data.get("benchmark") or {}
    if not isinstance(benchmark_data, Mapping):
        raise ValueError("benchmark must be a YAML mapping")
    benchmark = BenchmarkConfig(
        num_requests=_positive_int(
            benchmark_data.get("num_requests", 100), "benchmark.num_requests"
        ),
        warmup_requests=_positive_int(
            benchmark_data.get("warmup_requests", 0),
            "benchmark.warmup_requests",
            allow_zero=True,
        ),
        max_concurrency=_positive_int(
            benchmark_data.get("max_concurrency", maximum),
            "benchmark.max_concurrency",
        ),
        request_rate=_request_rate(benchmark_data.get("request_rate", "inf")),
        seed=_integer(benchmark_data.get("seed", 42), "benchmark.seed"),
    )

    condition_data = data.get("conditions")
    if not isinstance(condition_data, list) or not condition_data:
        raise ValueError("conditions must be a non-empty YAML list")
    conditions = []
    for index, item in enumerate(condition_data):
        if not isinstance(item, Mapping):
            raise ValueError(f"conditions[{index}] must be a YAML mapping")
        metric = item.get("metric")
        comparison = item.get("operator")
        value = item.get("value")
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"conditions[{index}].metric is not supported: {metric}")
        if comparison not in _OPERATORS:
            raise ValueError(f"conditions[{index}].operator is not supported")
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"conditions[{index}].value must be numeric")
        conditions.append(Condition(metric, comparison, float(value)))

    return SLOConfig(endpoint, request, benchmark, search, conditions)


def load_slo_config(path: str | Path) -> SLOConfig:
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise ValueError("SLO config must be a YAML mapping")
    return parse_slo_config(data)


def _build_payloads(config: SLOConfig, count: int, seed_offset: int = 0) -> List[Dict]:
    if config.request.payload is not None:
        return [dict(config.request.payload) for _ in range(count)]

    randomizer = random.Random(config.benchmark.seed + seed_offset)
    return [
        {
            "prompt": randomizer.choices(range(10_000), k=config.request.input_len),
            "max_tokens": config.request.output_len,
            "ignore_eos": True,
        }
        for _ in range(count)
    ]


def _request_url(config: SLOConfig) -> str:
    if config.endpoint.path:
        return api_url(config.endpoint.base_url, config.endpoint.path)
    payload = config.request.payload or {"prompt": []}
    endpoint = (
        "/v1/completions"
        if "prompt" in payload and "messages" not in payload
        else "/v1/chat/completions"
    )
    return api_url(config.endpoint.base_url, endpoint)


async def _send_requests(
    session,
    config: SLOConfig,
    payloads: List[Dict],
    max_concurrency: int,
    request_rate: float,
) -> List[OutputMetric]:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    async for payload in get_request(payloads, request_rate):
        prepared = prepare_payload(payload, config.endpoint.model)
        tasks.append(
            asyncio.create_task(
                request_func(
                    session,
                    _request_url(config),
                    prepared,
                    sem=semaphore,
                )
            )
        )
    return await asyncio.gather(*tasks)


def summarize_outputs(
    outputs: List[OutputMetric],
    duration_s: float,
    max_concurrency: int,
    request_rate: float,
) -> Dict[str, int | float]:
    successful = [output for output in outputs if output.success]
    duration_s = max(duration_s, 1e-9)
    metrics: Dict[str, int | float] = {
        "total_requests": len(outputs),
        "successful_requests": len(successful),
        "failed_requests": len(outputs) - len(successful),
        "success_rate": len(successful) / len(outputs) if outputs else 0.0,
        "duration_s": duration_s,
        "request_throughput": len(successful) / duration_s,
        "output_throughput": (
            sum(output.completion_tokens for output in successful) / duration_s
        ),
        "max_concurrency": max_concurrency,
        "request_rate": request_rate,
    }

    tpot_values = [
        value
        for output in successful
        if (value := output.calculate_tpot_ms()) is not None
    ]
    series = {
        "ttft_ms": [output.ttft_ms for output in successful if output.ttft_ms > 0],
        "tpot_ms": tpot_values,
        # request_func exposes TPOT rather than a separate per-token ITL stream.
        "itl_ms": tpot_values,
        "latency_ms": [output.latency_ms for output in successful],
        "e2e_latency_ms": [output.latency_ms for output in successful],
        **{
            field: [getattr(output, field) for output in successful]
            for field in _SERIES
            if field
            not in {
                "ttft_ms",
                "tpot_ms",
                "itl_ms",
                "latency_ms",
                "e2e_latency_ms",
            }
        },
    }
    for name, values in series.items():
        if not values:
            continue
        metrics[f"mean_{name}"] = float(np.mean(values))
        for percentile in (50, 95, 99):
            metrics[f"p{percentile}_{name}"] = float(np.percentile(values, percentile))
    return metrics


def evaluate_conditions(
    metrics: Mapping[str, int | float], conditions: Sequence[Condition]
) -> tuple[bool, List[Dict[str, Any]]]:
    evaluations = []
    for condition in conditions:
        actual = metrics.get(condition.metric)
        passed = actual is not None and _OPERATORS[condition.operator](
            actual, condition.value
        )
        evaluations.append(
            {
                **asdict(condition),
                "actual": actual,
                "passed": bool(passed),
            }
        )
    return all(item["passed"] for item in evaluations), evaluations


async def _run_probe(config: SLOConfig, candidate: int) -> Dict[str, int | float]:
    if config.search.parameter == "max_concurrency":
        max_concurrency = candidate
        request_rate = config.benchmark.request_rate
    else:
        max_concurrency = config.benchmark.max_concurrency
        request_rate = float(candidate)

    async with _create_bench_client_session(
        max_concurrency, config.endpoint.api_key
    ) as session:
        if config.benchmark.warmup_requests:
            await _send_requests(
                session,
                config,
                _build_payloads(config, config.benchmark.warmup_requests, 1),
                max_concurrency,
                float("inf"),
            )

        start_time = time.perf_counter()
        outputs = await _send_requests(
            session,
            config,
            _build_payloads(config, config.benchmark.num_requests),
            max_concurrency,
            request_rate,
        )
        duration_s = time.perf_counter() - start_time

    return summarize_outputs(outputs, duration_s, max_concurrency, request_rate)


async def run_slo(config: SLOConfig) -> Dict[str, Any]:
    np.random.seed(config.benchmark.seed)
    left, right = config.search.minimum, config.search.maximum
    best = None
    runs = []
    while left <= right:
        candidate = (left + right) // 2
        metrics = await _run_probe(config, candidate)
        passed, evaluations = evaluate_conditions(metrics, config.conditions)
        runs.append(
            {
                "candidate": candidate,
                "passed": passed,
                "metrics": metrics,
                "conditions": evaluations,
            }
        )
        logger.info(
            "SLO probe %s=%s: %s",
            config.search.parameter,
            candidate,
            "passed" if passed else "failed",
        )
        if passed:
            best = candidate
            left = candidate + 1
        else:
            right = candidate - 1

    return {
        "status": "satisfied" if best is not None else "unsatisfied",
        "search_parameter": config.search.parameter,
        "best_value": best,
        "runs": runs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aib slo", description="Find the highest load satisfying YAML SLOs"
    )
    parser.add_argument("config", type=Path, help="SLO YAML configuration")
    parser.add_argument("-o", "--output", type=Path, help="Write result YAML")
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(run_slo(load_slo_config(args.config)))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))

    rendered = yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")
    return 0


__all__ = [
    "Condition",
    "SLOConfig",
    "evaluate_conditions",
    "load_slo_config",
    "run_slo",
    "summarize_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
