import copy
import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def resolve_config_path(config_path: str) -> str:
    candidate = Path(config_path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    path = (
        resources.files("ai_infra_bench.check.correctness.eval_dataset") / config_path
    )
    return str(path)


def generate_payload(
    prompt_template: str, row: Dict, default_payload: Dict, override_payload
):
    return generate_payload_from_content(
        prompt_template.format(**row), default_payload, override_payload
    )


def generate_payload_from_content(
    prompt_content: str, default_payload: Dict, override_payload
):
    user_message = {"role": "user", "content": prompt_content}
    payload = copy.deepcopy(default_payload)
    payload.update(override_payload)
    if payload.get("messages") is None:
        payload["messages"] = []
    payload["messages"].append(user_message)
    return payload


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def extract_response_text(response_json: Dict[str, Any]) -> str:
    choice = (response_json.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content") or ""
    if not content:
        logger.warning(
            f"content is None, the full response choice is\n{json.dumps(choice, ensure_ascii=False, indent=4)}"
        )
    return f"{content}"
