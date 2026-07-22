import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OutputMetric:
    payload: Dict = field(default_factory=dict)
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    success: bool = False
    out_text: str = ""
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None

    def update_stream_output(self, text: str, start_time: float):
        if not text:
            return

        self.out_text += text
        if self.ttft_ms == 0.0:
            self.ttft_ms = (time.perf_counter() - start_time) * 1000

    def handle_usage_data(self, usage_data):
        self.prompt_tokens = usage_data.get("prompt_tokens", 0)
        self.completion_tokens = usage_data.get("completion_tokens", 0)

        cached_tokens = usage_data.get("cached_tokens")
        if cached_tokens is not None:
            self.cached_tokens = int(cached_tokens)

        prompt_tokens_details = usage_data.get("prompt_tokens_details") or {}
        if isinstance(prompt_tokens_details, dict):
            cached_tokens = prompt_tokens_details.get("cached_tokens")
            if cached_tokens is not None:
                self.cached_tokens = int(cached_tokens)

    def stream_text_or_empty(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
