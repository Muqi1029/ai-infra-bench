import copy
import json
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf nodes to plain Python so aiohttp/json can serialize."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def generate_payload(
    prompt_template: str, row: Dict, default_payload: Dict, override_payload
):
    user_message = {"role": "user", "content": prompt_template.format(**row)}
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
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    return f"{reasoning}{content}"
