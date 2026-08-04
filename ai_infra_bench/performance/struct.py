import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    spec_correct_drafts_histogram: Optional[List] = None

    def update_stream_output(self, text: str, start_time: float):
        if not text:
            return

        self.out_text += text
        if self.ttft_ms == 0.0:
            self.ttft_ms = (time.perf_counter() - start_time) * 1000

    def handle_usage_data(self, usage_data):
        self.prompt_tokens = usage_data.get("prompt_tokens", 0)
        self.completion_tokens = usage_data.get("completion_tokens", 0)
        self.reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        cached_tokens = usage_data.get("cached_tokens")
        if cached_tokens is not None:
            self.cached_tokens = int(cached_tokens)

        prompt_tokens_details = usage_data.get("prompt_tokens_details") or {}
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = prompt_tokens_details.get("cached_tokens")
            if cached_tokens is not None:
                self.cached_tokens = int(cached_tokens)

    def handle_sglext_data(self, sglext):
        if cached_tokens_details := sglext.get("cached_tokens_details"):
            self.cached_tokens_device = cached_tokens_details.get("device", 0)
            self.cached_tokens_host = cached_tokens_details.get("host", 0)
        if spec_tokens_details := sglext.get("spec_tokens_details"):
            self.spec_num_proposed_drafts = spec_tokens_details.get(
                "spec_num_proposed_drafts", 0
            )
            self.spec_num_correct_drafts = spec_tokens_details.get(
                "spec_num_correct_drafts", 0
            )
            self.spec_accept_length = spec_tokens_details.get("spec_accept_length", 0)
            self.spec_accept_rate = spec_tokens_details.get("spec_accept_rate", 0)
            self.spec_verify_ct = spec_tokens_details.get("spec_verify_ct", 0)
            self.spec_correct_drafts_histogram = spec_tokens_details.get(
                "spec_correct_drafts_histogram", []
            )

    def stream_text_or_empty(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
