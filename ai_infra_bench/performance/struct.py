import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping

from ai_infra_bench.utils.draw import Color, color_print
from ai_infra_bench.utils.req import extract_response_metrics


class TextType(Enum):
    REASONING = auto()
    CONTENT = auto()
    TOOL_CALLS = auto()


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
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "spec_num_proposed_drafts",
            "spec_num_correct_drafts",
            "spec_accept_length",
            "spec_accept_rate",
            "spec_verify_ct",
            "spec_correct_drafts_histogram",
        ):
            value = metrics.get(field_name)
            if value is not None:
                setattr(self, field_name, value)

        cached_details = metrics.get("cached_tokens_details") or {}
        self.cached_tokens_device = cached_details.get(
            "device", self.cached_tokens_device
        )
        self.cached_tokens_host = cached_details.get("host", self.cached_tokens_host)

    def handle_usage_data(self, usage_data: Mapping[str, Any]) -> None:
        self.update_response_metrics({"usage": usage_data})

    def handle_sglext_data(self, sglext: Mapping[str, Any]) -> None:
        self.update_response_metrics({"sglext": sglext})

    def calculate_tpot_ms(self):
        if self.completion_tokens <= 1 or self.ttft_ms <= 0.0:
            return None

        generation_time_ms = self.latency_ms - self.ttft_ms
        if generation_time_ms < 0.0:
            return None
        self.tpot_ms = generation_time_ms / (self.completion_tokens - 1)
        return self.tpot_ms
