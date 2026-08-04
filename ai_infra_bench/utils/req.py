from typing import Dict

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


def normalize_payload(payload: Dict) -> None:
    """Align recorded SGLang request bodies with router/OpenAI expectations."""
    if payload.get("min_tokens") is not None and payload["min_tokens"] < 1:
        payload.pop("min_tokens")

    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return payload
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return payload
    # SGLang logs use schema_; router deserializer requires schema (OpenAI shape).
    if "schema" not in json_schema and "schema_" in json_schema:
        json_schema["schema"] = json_schema.pop("schema_")
    return payload


def sanitize_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        return f"http://{url.strip()}"
    if url.endswith("/v1"):
        url = url.rstrip("/v1")
    return url
