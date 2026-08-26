import json
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

STREAM_RETURN_PAYLOAD = {
    "stream": True,
    "stream_options": {
        "include_usage": True,
        "continuous_usage_stats": True,
    },
    "return_cached_tokens_details": True,
    "return_spec_tokens_details": True,
}

NO_STREAM_RETURN_PAYLOAD = {"stream": False, "return_meta_info": True}

USAGE_METRIC_KEYS = (
    "prompt_tokens",
    "reasoning_tokens",
    "completion_tokens",
)
CACHED_METRIC_KEYS = ("cached_tokens", "cached_tokens_details")
SPEC_METRIC_KEYS = (
    "spec_accept_rate",
    "spec_accept_length",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
    "spec_correct_drafts_histogram",
    "spec_cap_length",
    "spec_block_accept_length",
    "spec_cap_lens_histogram",
)


def tool_filter_request(payload: Mapping[str, Any]) -> bool:
    """Return whether a request can run without constrained decoding."""
    tool_choice = payload.get("tool_choice", payload.get("tool_choices"))
    if tool_choice == "required" or isinstance(tool_choice, dict):
        return False
    if any(
        tool.get("strict") or (tool.get("function") or {}).get("strict")
        for tool in payload.get("tools") or []
    ):
        return False
    return not bool(payload.get("response_format"))


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Align recorded request bodies with the OpenAI request shape."""
    if payload.get("min_tokens") is not None and payload["min_tokens"] < 1:
        payload.pop("min_tokens")

    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return payload
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return payload
    # Some recorded payloads use schema_; routers expect the OpenAI ``schema`` key.
    if "schema" not in json_schema and "schema_" in json_schema:
        json_schema["schema"] = json_schema.pop("schema_")
    return payload


def parse_override_payload(override_payload: str) -> Dict[str, Any]:
    try:
        override = json.loads(override_payload)
    except json.JSONDecodeError as error:
        raise ValueError("--override-payload must be valid JSON") from error
    if not isinstance(override, dict):
        raise ValueError("--override-payload must be a JSON object")
    return override


def prepare_payload(
    payload: Mapping[str, Any],
    model: Optional[str] = None,
    override_payload: Optional[str] = None,
    stream: Optional[bool] = None,
) -> Dict[str, Any]:
    """Copy and normalize a payload before request-specific fields are added."""
    prepared = normalize_payload(deepcopy(dict(payload)))
    if model:
        prepared["model"] = model

    if override_payload is not None:
        prepared.update(parse_override_payload(override_payload))

    if stream is not None:
        if stream:
            prepared.pop("return_meta_info", None)
            payload_extra = STREAM_RETURN_PAYLOAD
        else:
            for key in (
                "stream_options",
                "return_cached_tokens_details",
                "return_spec_tokens_details",
            ):
                prepared.pop(key, None)
            payload_extra = NO_STREAM_RETURN_PAYLOAD
        prepared.update(deepcopy(payload_extra))
    return prepared


def sanitize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def api_url(base_url: str, endpoint: str) -> str:
    return f"{sanitize_url(base_url)}/{endpoint.lstrip('/')}"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_value(keys: Sequence[str], sources: Iterable[Mapping[str, Any]]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def extract_response_metrics(response: Any) -> Dict[str, Any]:
    """Extract standard and extension metrics from a response or stream chunk."""
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(data, Mapping):
        return {}

    usage = _as_mapping(data.get("usage"))
    prompt_details = _as_mapping(usage.get("prompt_tokens_details"))
    completion_details = _as_mapping(usage.get("completion_tokens_details"))

    sglext = _as_mapping(data.get("sglext"))
    spec_details = _as_mapping(sglext.get("spec_tokens_details"))

    choices = data.get("choices") or []
    first_choice = choices[0] if choices else {}
    choice_meta = _as_mapping(_as_mapping(first_choice).get("meta_info"))

    metrics = {
        key: _first_value((key,), (usage, choice_meta))
        for key in ("prompt_tokens", "completion_tokens")
    }
    metrics["reasoning_tokens"] = _first_value(
        ("reasoning_tokens", "reasoning_token"),
        (usage, completion_details, choice_meta),
    )
    metrics.update(
        {
            "cached_tokens": _first_value(
                ("cached_tokens",), (prompt_details, choice_meta)
            ),
            "cached_tokens_details": _first_value(
                ("cached_tokens_details",), (sglext, choice_meta)
            ),
        }
    )
    metrics.update(
        {
            key: _first_value((key,), (spec_details, choice_meta))
            for key in SPEC_METRIC_KEYS
        }
    )
    return metrics


def update_metrics(metrics: Dict[str, Any], new_metrics: Mapping[str, Any]) -> None:
    metrics.update(
        {key: value for key, value in new_metrics.items() if value is not None}
    )


def add_common_args(parser: Namespace):
    parser.add_argument(
        "--base-url",
        default="127.0.0.1:30000",
        type=sanitize_url,
        help="The base URL of the endpoint service",
    )
    parser.add_argument(
        "--api-key", default="JustKeepMe", help="The API key of the endpoint service"
    )
    parser.add_argument("--model", type=str, help="The model to request")

    parser.add_argument(
        "--override-payload",
        type=str,
        help="Override request fields with a JSON object",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random request generation seed"
    )
