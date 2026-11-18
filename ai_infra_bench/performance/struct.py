import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping

from ai_infra_bench.utils.color import Color, color_print
from ai_infra_bench.utils.req import (
    SPEC_METRIC_KEYS,
    USAGE_METRIC_KEYS,
    extract_response_metrics,
)
from ai_infra_bench.utils.table import (
    format_histogram_percentages,
    format_mean,
    format_percentile,
    mean_value,
    percentile_value,
)


class TextType(Enum):
    REASONING = auto()
    CONTENT = auto()
    TOOL_CALLS = auto()


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"

    def __str__(self) -> str:
        return self.value


@dataclass
class OutputMetric:
    # input payload
    payload: Dict = field(default_factory=dict)

    # latency
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    tpot_ms: float = 0.0
    itl_ms: List[float] = field(default_factory=list)

    # status
    success: bool = False
    error_message: str | None = None

    # finish_reason
    finish_reason: str | None = None

    # messages
    content: str = ""
    reasoning_content: str = ""
    tool_calls: str = ""

    # usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    # cached tokens
    cached_tokens: int = 0
    cached_tokens_device: int = 0
    cached_tokens_host: int = 0

    # spec
    spec_num_proposed_drafts: int = 0
    spec_num_correct_drafts: int = 0

    spec_cap_length: int = 0
    spec_block_accept_length: int = 0
    spec_cap_lens_histogram: List[int] = field(default_factory=list)

    spec_accept_rate: float = 0.0
    spec_accept_length: float = 0.0
    spec_verify_ct: int = 0
    spec_correct_drafts_histogram: List[int] = field(default_factory=list)

    def update_output(
        self,
        text_or_tool_calls: Any,
        text_type: TextType,
        render_content: bool = False,
    ) -> None:
        if not text_or_tool_calls:
            return

        if text_type is TextType.REASONING:
            self.reasoning_content += text_or_tool_calls
            if render_content:
                color_print(text_or_tool_calls, Color.LIGHT_CYAN)
        elif text_type is TextType.CONTENT:
            self.content += text_or_tool_calls
            if render_content:
                color_print(text_or_tool_calls, Color.LIGHT_GREEN)

        elif text_type is TextType.TOOL_CALLS:
            tool_text_parts = []
            for tool_call in text_or_tool_calls:
                function = tool_call.get("function") or {}
                if func_name := function.get("name"):
                    tool_text_parts.append(f"Function={func_name}\nArgument:")
                    if render_content:
                        color_print(
                            f"\n\n[Tool Call Detected]: Function={func_name}\nArgument:",
                            Color.LIGHT_YELLOW,
                        )
                if func_arg := function.get("arguments"):
                    tool_text_parts.append(func_arg)
                    if render_content:
                        color_print(func_arg, Color.LIGHT_YELLOW)
            self.tool_calls += "".join(tool_text_parts)

    def update_stream_output(
        self,
        text_or_tool_calls: Any,
        start_time: float,
        text_type: TextType,
        render_content: bool = False,
    ) -> None:
        if not text_or_tool_calls:
            return

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if self.ttft_ms == 0.0:
            self.ttft_ms = elapsed_ms
        else:
            prev_ms = self.ttft_ms + sum(self.itl_ms)
            self.itl_ms.append(max(0.0, elapsed_ms - prev_ms))
        self.update_output(text_or_tool_calls, text_type, render_content)

    def update_non_stream_response(
        self, response: Mapping[str, Any], render_content: bool = False
    ) -> None:
        self.update_response_metrics(response)

        choices = response.get("choices") or []
        if not choices:
            return

        choice = choices[0]
        if finish_reason := choice.get("finish_reason"):
            self.finish_reason = finish_reason

        if "text" in choice:
            self.update_output(choice.get("text", ""), TextType.CONTENT, render_content)
            return

        message = choice.get("message") or {}
        self.update_output(
            message.get("reasoning_content", "") or message.get("reasoning", ""),
            TextType.REASONING,
            render_content,
        )
        self.update_output(message.get("content", ""), TextType.CONTENT, render_content)
        if tool_calls := message.get("tool_calls"):
            self.update_output(tool_calls, TextType.TOOL_CALLS, render_content)

    def update_response_metrics(self, response: Mapping[str, Any]) -> None:
        metrics = extract_response_metrics(response)
        for field_name in (
            *USAGE_METRIC_KEYS,
            "cached_tokens",
            *SPEC_METRIC_KEYS,
        ):
            value = metrics.get(field_name)
            if value is None:
                continue
            # Streaming chunks often repeat usage with cached_tokens=0 after
            # prefill. Keep the high-water mark so Global cache ratio is not
            # wiped by a later empty/zero chunk.
            if field_name == "cached_tokens":
                self.cached_tokens = max(self.cached_tokens, value)
            else:
                setattr(self, field_name, value)

        cached_details = metrics.get("cached_tokens_details") or {}
        device = cached_details.get("device")
        host = cached_details.get("host")
        if device is not None:
            self.cached_tokens_device = max(self.cached_tokens_device, device)
        if host is not None:
            self.cached_tokens_host = max(self.cached_tokens_host, host)
        if not self.cached_tokens:
            self.cached_tokens = self.cached_tokens_device + self.cached_tokens_host

    def calculate_tpot_ms(self):
        if self.completion_tokens <= 1 or self.ttft_ms <= 0.0:
            return None

        generation_time_ms = self.latency_ms - self.ttft_ms
        if generation_time_ms < 0.0:
            return None
        self.tpot_ms = generation_time_ms / (self.completion_tokens - 1)
        return self.tpot_ms


MetricRows = List[List[Any]]
MetricTable = tuple[str, MetricRows]


@dataclass
class _OutputStats:
    all_outputs: List[OutputMetric]
    duration_s: float
    max_concurrency: int | None
    request_rate: float
    label: str
    ts: str
    goodput_thresholds: Mapping[str, float] | None = None
    peak_concurrency: int | None = None
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
            (
                "ITL",
                [itl for output in self.outputs for itl in output.itl_ms],
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

    def agg_histogram(self, histogram: List[List[int]]) -> List[int]:
        max_histogram_length = max((len(o) for o in histogram), default=0)
        total_histogram = [
            sum(o[i] for o in histogram if i < len(o))
            for i in range(max_histogram_length)
        ]
        return total_histogram

    def build_spec_table(self) -> MetricTable | None:
        total_proposed_drafts = sum(
            output.spec_num_proposed_drafts for output in self.outputs
        )
        if not total_proposed_drafts:
            return None

        total_correct_drafts = sum(
            output.spec_num_correct_drafts for output in self.outputs
        )
        avg_spec_accept_rate = f"{total_correct_drafts / total_proposed_drafts:.2%}"
        spec_outputs = [output for output in self.outputs if output.spec_verify_ct]
        total_spec_verify_ct = sum(output.spec_verify_ct for output in spec_outputs)
        total_completion_tokens = sum(
            output.completion_tokens for output in spec_outputs
        )

        agg_spec_correct_drafts_histogram = self.agg_histogram(
            [output.spec_correct_drafts_histogram for output in self.outputs]
        )

        table = [
            "Spec Tokens Statistics",
            [
                ["Metric", "Value"],
                ["Total Proposed Drafts", total_proposed_drafts],
                ["Total Correct Drafts", total_correct_drafts],
                ["Total Verify Count", total_spec_verify_ct],
                ["Avg Spec Accept Rate(All Verify)", avg_spec_accept_rate],
                [
                    "Avg Spec Accept Length(All Verify)",
                    (
                        f"{total_completion_tokens / total_spec_verify_ct:.2f}"
                        if total_spec_verify_ct
                        else "N/A"
                    ),
                ],
                [
                    "Agg Spec Correct Drafts Histogram",
                    agg_spec_correct_drafts_histogram,
                ],
                [
                    "Spec Correct Drafts Histogram Percentages",
                    format_histogram_percentages(agg_spec_correct_drafts_histogram),
                ],
            ],
        ]

        total_spec_num_cap_tokens = sum(
            output.spec_cap_length * output.spec_verify_ct for output in self.outputs
        )
        total_cap_proposed_tokens = total_spec_num_cap_tokens - total_spec_verify_ct
        if not total_spec_num_cap_tokens:
            return table

        agg_spec_cap_lens_histogram = self.agg_histogram(
            [output.spec_cap_lens_histogram for output in self.outputs]
        )

        cap_info = [
            [
                "Avg Cap Verify Len",
                f"{total_spec_num_cap_tokens / total_spec_verify_ct:.2f} ({total_proposed_drafts / total_spec_verify_ct + 1})",
            ],
            [
                "Avg Cap Spec Accept Rate",
                f"{total_correct_drafts / total_cap_proposed_tokens:.2%}",
            ],
            [
                "Agg Spec Cap Verify Len Histogram",
                agg_spec_cap_lens_histogram,
            ],
            [
                "Spec Cap Verify Len Histogram Percentages",
                format_histogram_percentages(agg_spec_cap_lens_histogram),
            ],
        ]
        table[1] += cap_info
        return table

    def _meets_goodput(self, output: OutputMetric) -> bool:
        if not self.goodput_thresholds:
            return True
        for metric, limit in self.goodput_thresholds.items():
            if metric == "ttft":
                if output.ttft_ms <= 0 or output.ttft_ms > limit:
                    return False
            elif metric == "tpot":
                tpot = output.calculate_tpot_ms()
                if output.completion_tokens > 1 and (tpot is None or tpot > limit):
                    return False
                if tpot is not None and tpot > limit:
                    return False
            elif metric == "itl":
                if any(val > limit for val in output.itl_ms):
                    return False
            elif metric == "e2el":
                if output.latency_ms > limit:
                    return False
        return True

    def build_finish_reason_table(self):
        counts: Dict[str, int] = {str(reason): 0 for reason in FinishReason}
        for output in self.outputs:
            key = str(output.finish_reason) if output.finish_reason else "none"
            counts[key] = counts.get(key, 0) + 1
        if counts.get("none", 0) == 0:
            counts.pop("none", None)
        return (
            "Finish Reason Statistics",
            [
                ["Finish reason", "Requests", "Percentage"],
                *[
                    [
                        finish_reason,
                        str(count),
                        f"{count / self.num_success_requests:.2%}",
                    ]
                    for finish_reason, count in counts.items()
                ],
            ],
        )

    def build_benchmark_summary_table(self):
        total_prompt_tokens = sum(output.prompt_tokens for output in self.outputs)
        total_completion_tokens = sum(
            output.completion_tokens for output in self.outputs
        )
        total_reasoning_tokens = sum(output.reasoning_tokens for output in self.outputs)
        total_cached_tokens = sum(output.cached_tokens for output in self.outputs)
        total_cached_tokens_device = sum(
            output.cached_tokens_device for output in self.outputs
        )
        total_cached_tokens_host = sum(
            output.cached_tokens_host for output in self.outputs
        )
        cached_tokens_device_ratio = (
            total_cached_tokens_device / total_cached_tokens
            if total_cached_tokens
            else 0.0
        )
        cached_tokens_host_ratio = (
            total_cached_tokens_host / total_cached_tokens
            if total_cached_tokens
            else 0.0
        )
        total_cacheable_prompt_tokens = total_prompt_tokens - self.num_success_requests
        global_cache_ratio = (
            total_cached_tokens / total_cacheable_prompt_tokens
            if total_cacheable_prompt_tokens > 0
            else 0.0
        )
        request_rate_display = (
            "unlimited"
            if self.request_rate == float("inf")
            else f"{self.request_rate:g} req/s"
        )
        if self.max_concurrency is None:
            concurrency_display = (
                f"unlimited (peak: {self.peak_concurrency})"
                if self.peak_concurrency is not None
                else "unlimited"
            )
        else:
            concurrency_display = str(self.max_concurrency)
        summary_context = " ".join(value for value in (self.label, self.ts) if value)
        rows = [
            ["Metric", "Value"],
            ["Total requests", str(self.num_total_requests)],
            ["Successful requests", str(self.num_success_requests)],
            ["Failed requests", str(self.num_failed_requests)],
            ["Max concurrency", concurrency_display],
            ["Request rate", request_rate_display],
            [
                "Request Throughput",
                f"{self.num_success_requests / self.duration_s:.2f} req/s",
            ],
        ]
        if self.goodput_thresholds:
            good_count = sum(
                1 for output in self.outputs if self._meets_goodput(output)
            )
            rows.append(
                [
                    "Request goodput",
                    f"{good_count / self.duration_s:.2f} req/s ({good_count / max(self.num_total_requests, 1):.2%})",
                ]
            )
        rows.extend(
            [
                ["Duration", f"{self.duration_s:.2f} s"],
                [
                    "Input throughput",
                    f"{total_prompt_tokens / self.duration_s:.2f} tokens/s",
                ],
                [
                    "Output throughput",
                    f"{total_completion_tokens / self.duration_s:.2f} tokens/s",
                ],
                [
                    "Total token throughput",
                    f"{(total_prompt_tokens + total_completion_tokens) / self.duration_s:.2f} tokens/s",
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
            ]
        )
        return (f"Benchmark Summary ({summary_context})", rows)

    def build_latency_token_table(self):
        ratios = self.cached_token_ratios
        ratio_mean = mean_value(ratios)
        ratio_p50 = percentile_value(ratios, 50)
        ratio_p95 = percentile_value(ratios, 95)
        ratio_p99 = percentile_value(ratios, 99)
        return (
            "Latency & Token Metrics",
            [
                ["Metric", "Mean", "P50", "P95", "P99", "Unit"],
                *[
                    [metric, *self.compute_metrics(values), unit]
                    for metric, values, unit in self.metric_series()
                ],
                [
                    "Cached token ratio",
                    f"{ratio_mean:.2%}" if ratio_mean is not None else "N/A",
                    f"{ratio_p50:.2%}" if ratio_p50 is not None else "N/A",
                    f"{ratio_p95:.2%}" if ratio_p95 is not None else "N/A",
                    f"{ratio_p99:.2%}" if ratio_p99 is not None else "N/A",
                    "ratio",
                ],
            ],
        )

    @staticmethod
    def compute_metrics(values: List[float | int]) -> List[str]:
        return [
            format_mean(values),
            format_percentile(values, 50),
            format_percentile(values, 95),
            format_percentile(values, 99),
        ]

    @staticmethod
    def _stat_dict(values: List[float | int], scale: float = 1.0) -> Dict[str, float]:
        avg = mean_value(values)
        p50 = percentile_value(values, 50)
        p95 = percentile_value(values, 95)
        p99 = percentile_value(values, 99)
        if avg is None or p50 is None or p95 is None or p99 is None:
            return {}
        return {
            "mean": round(avg * scale, 2),
            "p50": round(p50 * scale, 2),
            "p95": round(p95 * scale, 2),
            "p99": round(p99 * scale, 2),
        }

    def build_yaml_record(self) -> Dict[str, Any]:
        is_qps_mode = self.request_rate != float("inf") and self.max_concurrency is None
        record: Dict[str, Any] = {
            "label": self.label or "default",
            "timestamp": self.ts,
            "mode": "qps" if is_qps_mode else "concurrency",
            "max_concurrency": (
                "unlimited" if self.max_concurrency is None else self.max_concurrency
            ),
        }
        if self.peak_concurrency is not None:
            record["peak_concurrency"] = self.peak_concurrency
        record["request_rate"] = (
            "unlimited"
            if self.request_rate == float("inf")
            else round(float(self.request_rate), 4)
        )

        if not self.outputs:
            record["summary"] = {
                "total_requests": self.num_total_requests,
                "successful_requests": 0,
                "failed_requests": self.num_failed_requests,
                "status": "No successful requests",
            }
            return record

        total_prompt = sum(o.prompt_tokens for o in self.outputs)
        total_completion = sum(o.completion_tokens for o in self.outputs)
        total_reasoning = sum(o.reasoning_tokens for o in self.outputs)
        total_cached = sum(o.cached_tokens for o in self.outputs)
        total_cached_dev = sum(o.cached_tokens_device for o in self.outputs)
        total_cached_host = sum(o.cached_tokens_host for o in self.outputs)
        cacheable_prompt = total_prompt - self.num_success_requests
        global_cache_pct = (
            round(100.0 * total_cached / cacheable_prompt, 2)
            if cacheable_prompt > 0
            else 0.0
        )

        summary: Dict[str, Any] = {
            "total_requests": self.num_total_requests,
            "successful_requests": self.num_success_requests,
            "failed_requests": self.num_failed_requests,
            "duration_s": round(self.duration_s, 4),
            "request_throughput_rps": round(
                self.num_success_requests / self.duration_s, 4
            ),
        }
        if self.goodput_thresholds:
            good_count = sum(1 for o in self.outputs if self._meets_goodput(o))
            summary["request_goodput_rps"] = round(good_count / self.duration_s, 4)
            summary["request_goodput_pct"] = round(
                100.0 * good_count / max(self.num_total_requests, 1), 2
            )
        summary.update(
            {
                "input_throughput_tps": round(total_prompt / self.duration_s, 2),
                "output_throughput_tps": round(total_completion / self.duration_s, 2),
                "total_token_throughput_tps": round(
                    (total_prompt + total_completion) / self.duration_s, 2
                ),
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_reasoning_tokens": total_reasoning,
                "total_cached_tokens": total_cached,
                "global_cache_ratio_pct": global_cache_pct,
            }
        )
        if total_cached_dev or total_cached_host:
            summary["cached_tokens_device"] = total_cached_dev
            summary["cached_tokens_host"] = total_cached_host
        record["summary"] = summary

        key_map = {
            "TTFT": "ttft_ms",
            "TPOT(ecl the ttft)": "tpot_ms",
            "ITL": "itl_ms",
            "Latency": "latency_ms",
            "Prompt tokens": "prompt_tokens",
            "Reasoning tokens": "reasoning_tokens",
            "Cached tokens": "cached_tokens",
            "Completion tokens": "completion_tokens",
        }
        metrics_section: Dict[str, Dict[str, float]] = {}
        for title, values, _ in self.metric_series():
            yaml_key = key_map.get(title, title.lower().replace(" ", "_"))
            if yaml_key in {"reasoning_tokens", "cached_tokens"} and not any(values):
                continue
            stat = self._stat_dict(values)
            if stat:
                metrics_section[yaml_key] = stat
        if any(self.cached_token_ratios):
            ratio_stat = self._stat_dict(self.cached_token_ratios, scale=100.0)
            if ratio_stat:
                metrics_section["cached_token_ratio_pct"] = ratio_stat
        record["metrics"] = metrics_section

        total_proposed = sum(o.spec_num_proposed_drafts for o in self.outputs)
        if total_proposed > 0:
            total_correct = sum(o.spec_num_correct_drafts for o in self.outputs)
            spec_outputs = [o for o in self.outputs if o.spec_verify_ct]
            total_verify = sum(o.spec_verify_ct for o in spec_outputs)
            spec_comp = sum(o.completion_tokens for o in spec_outputs)
            hist = self.agg_histogram(
                [o.spec_correct_drafts_histogram for o in self.outputs]
            )
            record["spec_tokens"] = {
                "total_proposed_drafts": total_proposed,
                "total_correct_drafts": total_correct,
                "total_verify_count": total_verify,
                "accept_rate_pct": round(100.0 * total_correct / total_proposed, 2),
                "accept_length": (
                    round(spec_comp / total_verify, 2) if total_verify else None
                ),
                "correct_drafts_histogram": hist,
            }

        finish_counts: Dict[str, int] = {}
        for o in self.outputs:
            reason = str(o.finish_reason) if o.finish_reason else "none"
            finish_counts[reason] = finish_counts.get(reason, 0) + 1
        record["finish_reasons"] = {
            reason: {
                "requests": count,
                "pct": round(100.0 * count / self.num_success_requests, 2),
            }
            for reason, count in finish_counts.items()
            if count > 0
        }
        return record
