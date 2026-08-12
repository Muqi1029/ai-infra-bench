import argparse
import json
import random
import time
from typing import Any, Dict, Optional, Tuple

import requests

from ai_infra_bench.utils.device import get_first_gpu_info
from ai_infra_bench.utils.draw import Color, color_print, fmt, print_table
from ai_infra_bench.utils.ori import read_json
from ai_infra_bench.utils.req import (
    CACHED_METRIC_KEYS,
    NO_STREAM_RETURN_PAYLOAD,
    SPEC_METRIC_KEYS,
    STREAM_RETURN_PAYLOAD,
    USAGE_METRIC_KEYS,
    api_url,
    extract_response_metrics,
    format_histogram_percentages,
    prepare_payload,
    sanitize_url,
    update_metrics,
)

RANDOM_TOKEN_UPPER_BOUND = 10_000
json_schema_response_format = {
    "name": "require_named",
    "description": "a schema for the response format",
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "the name",
                "enum": ["Muqi Li", "Muqi1029"],
            },
            "age": {
                "type": "integer",
                "description": "a number from 0 to 23, which represent the person's age",
                "minimum": 0,
                "maximum": 23,
            },
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    },
    "strict": True,
}

tool_select_name = {
    "type": "function",
    "function": {
        "name": "select_name",
        "description": "select a name",
        "additionalproperties": False,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "the name",
                    "enum": ["Muqi Li", "Muqi1029"],
                },
                "age": {
                    "type": "integer",
                    "description": "a number from 0 to 23, which represent the person's age",
                    "minimum": 0,
                    "maximum": 23,
                },
            },
        },
        "required": ["name", "age"],
    },
}

tools = [tool_select_name]


ebnf_content = """
root ::= city | description

city ::= "London" | "Pairis" | "Berlin" | "Rome"
description ::= city " is " status

status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""


def info_print(headers, payload, url):
    print("=" * 80)
    print(f"Sending to {url=}")
    print(f"headers={json.dumps(headers, indent=2, ensure_ascii=False)}")
    print(f"payload={json.dumps(payload, indent=2, ensure_ascii=False)}")


def print_metrics(start_time, end_time, first_token_time=None, metrics=None):
    e2e = end_time - start_time
    ttft = first_token_time - start_time if first_token_time is not None else None
    e2e_ms = e2e * 1000
    ttft_ms = ttft * 1000 if ttft is not None else None

    metrics = metrics or {}
    completion_tokens = metrics.get("completion_tokens")

    token_per_sec = None
    if completion_tokens and e2e > 0:
        token_per_sec = completion_tokens / e2e

    tpot_ms = None
    if completion_tokens and completion_tokens > 1 and ttft_ms is not None:
        tpot_ms = (e2e_ms - ttft_ms) / (completion_tokens - 1)

    cached_tokens = metrics.get("cached_tokens") or 0
    prompt_tokens = metrics.get("prompt_tokens") or 0
    cached_tokens_ratio = (
        cached_tokens / (prompt_tokens - 1) if prompt_tokens > 0 else 0.0
    )
    device_name, device_memory = get_first_gpu_info()

    rows = [
        ("Metric", "Value", "Unit"),
        ("device info", device_name, ""),
        ("device memory", device_memory, ""),
        ("ttft", fmt(ttft_ms), "ms"),
        ("e2e latency", fmt(e2e_ms), "ms"),
        ("tpot", fmt(tpot_ms), "ms"),
        ("tps", fmt(token_per_sec), "tokens"),
        *[(key, fmt(metrics.get(key)), "tokens") for key in USAGE_METRIC_KEYS],
        ("cached_tokens", fmt(cached_tokens), "tokens"),
        ("cached_tokens_ratio", fmt(cached_tokens_ratio * 100), "%"),
    ]

    print_table(title="Single Request Benchmark", rows=rows)

    if cached_tokens:
        print_table(
            title="Cached Token Metrics",
            rows=[
                ("Metric", "Value"),
                *[(key, fmt(metrics.get(key))) for key in CACHED_METRIC_KEYS],
            ],
        )

    if metrics.get("spec_num_proposed_drafts"):
        print_table(
            title="Spec Token Metrics",
            rows=[
                ("Metric", "Value"),
                *[(key, fmt(metrics.get(key))) for key in SPEC_METRIC_KEYS],
                (
                    "spec_correct_drafts_histogram_percentages",
                    format_histogram_percentages(
                        metrics.get("spec_correct_drafts_histogram") or []
                    ),
                ),
            ],
        )


def build_request(args) -> Tuple[str, Dict[str, Any]]:
    url = api_url(args.base_url, "/v1/chat/completions")

    if args.payload_path:
        payload = read_json(args.payload_path)
    elif args.input_len is not None:
        input_ids = random.Random(args.seed).choices(
            range(RANDOM_TOKEN_UPPER_BOUND), k=args.input_len
        )
        payload = {
            "prompt": input_ids,
            "max_tokens": args.output_len,
            "ignore_eos": True,
        }
        url = api_url(args.base_url, "/v1/completions")
    elif args.input_ids_path:
        input_ids = read_json(args.input_ids_path)
        payload = {"prompt": input_ids}
        url = api_url(args.base_url, "/v1/completions")
    else:
        payload = {"messages": [{"role": "user", "content": args.prompt}]}
        if args.ebnf:
            payload["ebnf"] = ebnf_content
        elif args.json_schema_response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema_response_format,
            }
        elif args.tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"

    if args.disable_separate_reasoning:
        payload["separate_reasoning"] = False
    if args.enable_thinking:
        # for compatibility of different platforms
        payload["chat_template_kwargs"] = {
            "thinking": True,
            "enable_thinking": True,
        }
        payload["thinking"] = {"type": "enabled"}
    elif args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
        payload["thinking"] = {"type": "disabled"}

    return url, prepare_payload(payload, args.model)


def _print_request_error(response, error: Exception) -> None:
    color_print(
        f"Request Error, Status Code={response.status_code}, "
        f"Reason: {response.text} Error: {error}",
        Color.RED,
    )


def _handle_non_stream_request(url, headers, payload) -> None:
    payload.update(NO_STREAM_RETURN_PAYLOAD)
    start_time = time.perf_counter()
    response = requests.post(url=url, headers=headers, json=payload)
    end_time = time.perf_counter()

    try:
        response.raise_for_status()
        response_json = response.json()
        color_print(
            json.dumps(response_json, indent=2, ensure_ascii=False),
            Color.LIGHT_GREEN,
        )
        metrics = extract_response_metrics(response_json)
    except Exception as error:
        _print_request_error(response, error)
        metrics = None

    print_metrics(start_time, end_time, metrics=metrics)


def _render_delta(delta: Dict[str, Any]) -> bool:
    rendered_token = False
    for key, color in (
        ("reasoning_content", Color.LIGHT_CYAN),
        ("content", Color.LIGHT_GREEN),
    ):
        if text := delta.get(key):
            color_print(text, color)
            rendered_token = True

    for tool_call in delta.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if func_name := function.get("name"):
            color_print(
                f"\n\n[Tool Call Detected]: Function={func_name}\nArgument:",
                Color.LIGHT_YELLOW,
            )
            rendered_token = True
        if func_arg := function.get("arguments"):
            color_print(func_arg, Color.LIGHT_YELLOW)
            rendered_token = True
    return rendered_token


def _render_stream_choice(choice: Dict[str, Any], raw: bool) -> bool:
    if "text" in choice:
        text = choice.get("text") or ""
        if text and not raw:
            color_print(text, Color.LIGHT_GREEN)
        return bool(text)

    delta = choice.get("delta") or {}
    has_token = bool(
        delta.get("reasoning_content")
        or delta.get("content")
        or delta.get("tool_calls")
    )
    if not raw:
        has_token = _render_delta(delta)
    return has_token


def _handle_stream_request(url, headers, payload, raw: bool) -> None:
    payload.update(STREAM_RETURN_PAYLOAD)
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    metrics: Dict[str, Any] = {}

    try:
        response = requests.post(url=url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            decoded_line = line.decode("utf-8")
            if raw:
                print(decoded_line)
            if not decoded_line.startswith("data:"):
                continue

            data_str = decoded_line[len("data:") :].lstrip()
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            update_metrics(metrics, extract_response_metrics(chunk))
            choices = chunk.get("choices") or []
            received_at = time.perf_counter()
            has_token = _render_stream_choice(choices[0], raw) if choices else False
            if has_token and first_token_time is None:
                first_token_time = received_at

        end_time = time.perf_counter()
        print_metrics(start_time, end_time, first_token_time, metrics)
    except requests.HTTPError as error:
        response = error.response
        body = response.text if response is not None else ""
        color_print(
            f"Request Error, Status Code="
            f"{getattr(response, 'status_code', 'N/A')}, Reason: {body}",
            Color.RED,
        )
        raise
    except Exception as error:
        color_print(f"Request Error: {error}", Color.RED)
        raise


def http_request(args):
    url, payload = build_request(args)
    headers = {"Authorization": f"Bearer {args.api_key}"}

    if args.verbose:
        info_print(headers, payload, url)
    if args.disable_stream:
        _handle_non_stream_request(url, headers, payload)
    else:
        _handle_stream_request(url, headers, payload, args.raw)


def main(argv=None):
    parser = argparse.ArgumentParser("")
    parser.add_argument("--base-url", type=sanitize_url, default="localhost:8888")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--model", type=str, help="override the model field in the payload"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--disable-stream", action="store_true")
    parser.add_argument("--prompt", type=str, default="Who are you")
    parser.add_argument("--seed", type=int, default=42, help="Random token seed")
    parser.add_argument(
        "--output-len",
        type=int,
        help="Force this many output tokens with ignore_eos",
    )

    # extra kwargs in the payload
    think_mutex_group = parser.add_mutually_exclusive_group()
    think_mutex_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Whether to enable reasoning",
    )
    think_mutex_group.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Whether to disable reasoning",
    )

    parser.add_argument(
        "--disable-separate-reasoning",
        action="store_true",
        help="Whether to separate reasoning",
    )

    parser.add_argument(
        "--raw", action="store_true", help="Whether to print raw sse content"
    )

    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--ebnf", action="store_true", help="Constrained Decoding for EBNF format"
    )
    mutex_group.add_argument(
        "--json-schema-response-format",
        action="store_true",
        help="JSON Schema Response Format",
    )
    mutex_group.add_argument("--tools", action="store_true", help="Add tool")
    mutex_group.add_argument("--payload-path", type=str, help="The path of payload")
    mutex_group.add_argument("--input-ids-path", type=str, help="The path of input_ids")
    mutex_group.add_argument(
        "--input-len",
        type=int,
        help="Generate this many random input token IDs",
    )

    args = parser.parse_args(argv)
    if (args.input_len is None) != (args.output_len is None):
        parser.error("--input-len and --output-len must be used together")
    if args.input_len is not None and args.input_len < 1:
        parser.error("--input-len must be at least 1")
    if args.output_len is not None and args.output_len < 0:
        parser.error("--output-len must be at least 1")
    http_request(args)


if __name__ == "__main__":
    main()
