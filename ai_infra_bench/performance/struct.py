import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ai_infra_bench.utils.req import extract_response_metrics


@dataclass
class OutputMetric:
    payload: Dict = field(default_factory=dict)
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    success: bool = False
    out_text: str = ""
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

    def update_stream_output(self, text: str, start_time: float):
        if not text:
            return

        self.out_text += text
        if self.ttft_ms == 0.0:
            self.ttft_ms = (time.perf_counter() - start_time) * 1000

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

    def stream_text_or_empty(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
