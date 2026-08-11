import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Mapping, Optional

from ai_infra_bench.utils.req import extract_response_metrics


class TextType(Enum):
    REASONING = auto()
    CONTENT = auto()
    TOOL_CALLS = auto()


@dataclass
class OutputMetric:
    payload: Dict = field(default_factory=dict)
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    success: bool = False
    content: str = ""
    reasoning_content: str = ""
    tool_calls: str = ""
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None

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

    def update_stream_output(self, text: str, start_time: float, text_type: TextType):
        if not text:
            return

        if self.ttft_ms == 0.0:
            self.ttft_ms = (time.perf_counter() - start_time) * 1000

        if text_type is TextType.REASONING:
            self.reasoning_content += text
        elif text_type is TextType.CONTENT:
            self.content += text
        elif text_type is TextType.TOOL_CALLS:
            self.tool_calls += text

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
