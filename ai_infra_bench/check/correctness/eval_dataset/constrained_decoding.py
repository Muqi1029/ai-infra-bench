import copy
import json
from typing import Any, Dict, Tuple

from omegaconf import OmegaConf

from ai_infra_bench.check.correctness.eval_dataset.base import Eval
from ai_infra_bench.check.correctness.eval_dataset.utils import (
    generate_payload,
    resolve_config_path,
)

schema = {
    "type": "object",
    "properties": {
        "name_": {
            "type": "string",
            "description": "the name",
            "enum": ["Muqi Li", "Muqi1029"],
        },
        "_age": {
            "type": "integer",
            "description": "age between 0-24",
            "minimum": 0,
            "maximum": 24,
        },
    },
    "required": ["name_", "_age"],
}
tools = [
    {
        "type": "function",
        "function": {
            "name": "select_name",
            "description": "select a name",
            "parameters": schema,
        },
    }
]
strict_tools = [
    {
        "type": "function",
        "function": {
            "name": "select_name",
            "description": "select a name",
            "strict": True,
            "parameters": schema,
        },
    }
]
PROMPT = "Please invoke the tool to select a name. My name is Muqi Li, age is 24"
MODE_TO_EXTRA_PAYLOAD: Dict[str, Dict] = {
    "tool_choice_none": {"tool_choice": "none", "tools": tools},
    "tool_choice_auto": {"tool_choice": "auto", "tools": tools},
    "tool_choice_auto_strict": {"tool_choice": "auto", "tools": strict_tools},
    "tool_choice_fc": {
        "tool_choice": {"type": "function", "function": {"name": "select_name"}},
        "tools": tools,
    },
    "tool_choice_required": {"tool_choice": "required", "tools": tools},
    "response_format_json_schema": {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "select_name", "schema": schema},
        }
    },
}


def check_function(function) -> bool:
    if function is None:
        return False
    if function["name"] != "select_name":
        return False
    args = json.loads(function["arguments"])
    if args["name_"] != "Muqi Li" or args["_age"] != 24:
        return False
    return True


def check_answer(response_body, mode: str, is_thinking: bool) -> bool:
    choice = (response_body.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    has_reasoning_content = message.get("reasoning_content") is not None
    if (is_thinking and not has_reasoning_content) or (
        not is_thinking and has_reasoning_content
    ):
        return False

    tool_calls = message.get("tool_calls")
    has_tool_calls = tool_calls is not None and tool_calls

    if mode == "tool_choice_none":
        return not has_tool_calls
    elif mode == "tool_choice_auto":
        return True
    elif mode == "tool_choice_auto_strict":
        if has_tool_calls:
            return check_function(tool_calls[0].get("function"))
        return True
    elif mode in ["tool_choice_fc", "tool_choice_required"]:
        if not has_tool_calls:
            return False
        return check_function(tool_calls[0].get("function"))
    elif mode == "response_format_json_schema":
        data = json.loads(message.get("content"))
        if data["name_"] != "Muqi Li" or data["_age"] != 24:
            return False
        return True
    else:
        raise ValueError(f"{mode=} is not supported")


class ConstrainedDecodingEval(Eval):
    name: str = "Constrained Decoding"

    def __init__(self, name: str, config_path="configs/constrained_decoding.yaml"):
        self.name = name.replace("_", " ").title()
        self.results = []
        cfg = OmegaConf.load(resolve_config_path(config_path))

        self.prompt_template = cfg.get("prompt_template", "")
        self.default_payload = OmegaConf.to_container(
            cfg.get("payload", {}), resolve=True
        )

        self.rows = [
            {"question": PROMPT, "is_thinking_and_mode": (is_thinking, mode)}
            for is_thinking in [True, False]
            for mode in MODE_TO_EXTRA_PAYLOAD.keys()
        ]

    def maybe_truncate(self, num_questions: int | None):
        pass

    def get_length(self) -> int:
        return len(self.rows)

    def get_payload_and_answer(self, override_payload: Dict) -> Tuple[Dict, Any]:
        for row in self.rows:
            is_thinking, mode = row["is_thinking_and_mode"]
            # Fresh copies each request — never mutate shared MODE_TO_EXTRA_PAYLOAD
            # or the caller's override_payload (would leak fields across modes).
            mode_payload = copy.deepcopy(MODE_TO_EXTRA_PAYLOAD[mode])
            mode_payload["chat_template_kwargs"] = {
                "enable_thinking": is_thinking,
                "thinking": is_thinking,
            }
            merged_override_payload = copy.deepcopy(override_payload)
            merged_override_payload.update(mode_payload)

            payload = generate_payload(
                self.prompt_template,
                row,
                default_payload=self.default_payload,
                override_payload=merged_override_payload,
            )
            yield payload, row["is_thinking_and_mode"]

    def _eval(self, response_body, is_thinking_and_mode, payload=None) -> bool:
        is_thinking, mode = is_thinking_and_mode
        is_right = check_answer(response_body, mode, is_thinking=is_thinking)
        if not is_right:
            print(
                f"\033[42mThinking: {is_thinking}, Constrained Decoding Mode: {mode} FAILED\033[0m"
                f"\nResponse Body:{json.dumps(response_body, indent=2, ensure_ascii=False)}"
                f"\nPayload: {json.dumps(payload, indent=2, ensure_ascii=False)}"
            )
        return is_right
