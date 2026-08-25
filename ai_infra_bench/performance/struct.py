import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from typing import Any, Dict, List, Mapping

import numpy as np

from ai_infra_bench.utils.draw import (
    Color,
    color_print,
    format_histogram_percentages,
    format_mean,
    format_percentile,
)
from ai_infra_bench.utils.req import (
    SPEC_METRIC_KEYS,
    USAGE_METRIC_KEYS,
    extract_response_metrics,
)


class TextType(Enum):
    REASONING = auto()
    CONTENT = auto()
    TOOL_CALLS = auto()


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"


@dataclass
class OutputMetric:
    # input payload
    payload: Dict = field(default_factory=dict)

    # latency
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    tpot_ms: float = 0.0

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

        if self.ttft_ms == 0.0:
            self.ttft_ms = (time.perf_counter() - start_time) * 1000
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
            message.get("reasoning_content", ""),
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
            if value is not None:
                setattr(self, field_name, value)

        cached_details = metrics.get("cached_tokens_details") or {}
        self.cached_tokens_device = cached_details.get(
            "device", self.cached_tokens_device
        )
        self.cached_tokens_host = cached_details.get("host", self.cached_tokens_host)
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
                ["Toral Verify Count", total_spec_verify_ct],
                [
                    "Avg Spec Accept Rate(All Verify)",
                    f"{total_correct_drafts / total_proposed_drafts:.2%}",
                ],
                [
                    "Avg Spec Accept Length",
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

    def build_finish_reason_table(self):
        finish_reason_counts = {
            finish_reason: sum(
                output.finish_reason == finish_reason for output in self.outputs
            )
            for finish_reason in FinishReason
        }
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
                    for finish_reason, count in finish_reason_counts.items()
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
        return (
            "Benchmark Summary",
            [
                ["Metric", "Value"],
                ["Total requests", str(self.num_total_requests)],
                ["Successful requests", str(self.num_success_requests)],
                ["Failed requests", str(self.num_failed_requests)],
                ["Max concurrency", str(self.max_concurrency)],
                ["Request rate", request_rate_display],
                [
                    "Request Throughput",
                    f"{self.num_success_requests / self.duration_s:.2f} req/s",
                ],
                ["Duration", f"{self.duration_s:.2f} s"],
                [
                    "Output throughput",
                    f"{total_completion_tokens / self.duration_s:.2f} tokens/s",
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
        )

    def build_latency_token_table(self):
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
                    f"{np.mean(self.cached_token_ratios):.2%}",
                    f"{np.percentile(self.cached_token_ratios, 50):.2%}",
                    f"{np.percentile(self.cached_token_ratios, 95):.2%}",
                    f"{np.percentile(self.cached_token_ratios, 99):.2%}",
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
