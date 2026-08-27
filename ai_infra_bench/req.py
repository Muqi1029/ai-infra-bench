import argparse
import asyncio
import json
import random
import time
from typing import Any, Dict, Mapping, Tuple

import requests

from ai_infra_bench.performance.bench import read_packaged_requests
from ai_infra_bench.performance.bench_utils import handle_outputs
from ai_infra_bench.performance.core import request_func
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.draw import Color, color_print
from ai_infra_bench.utils.io import _read_json
from ai_infra_bench.utils.req import (
    add_common_args,
    api_url,
    parse_override_payload,
    prepare_payload,
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


def _select_dataset_request(dataset: str, seed: int) -> Dict[str, Any]:
    requests = read_packaged_requests(dataset)
    if not requests:
        raise ValueError(f"Dataset {dataset!r} does not contain any requests")
    payload = random.Random(seed).choice(requests)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Dataset {dataset!r} contains an invalid payload")
    return dict(payload)


def build_request(args) -> Tuple[str, Dict[str, Any]]:
    url = api_url(args.base_url, "/v1/chat/completions")

    if args.dataset:
        payload = _select_dataset_request(args.dataset, args.seed)
    elif args.payload_path:
        payload = _read_json(args.payload_path)
        if not isinstance(payload, Mapping):
            raise ValueError("--payload-path must contain a JSON object")
        if "prompt" in payload and "messages" not in payload:
            url = api_url(args.base_url, "/v1/completions")
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
        input_ids = _read_json(args.input_ids_path)
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

    prepared = prepare_payload(
        payload, args.model, args.override_payload, stream=not args.disable_stream
    )
    if "prompt" in prepared and "messages" not in prepared:
        url = api_url(args.base_url, "/v1/completions")
    return url, prepared


def _print_request_error(response, error: Exception) -> None:
    color_print(
        f"Request Error, Status Code={response.status_code}, "
        f"Reason: {response.text} Error: {error}",
        Color.RED,
    )


def _handle_request_output(output_metric: OutputMetric, duration_s: float) -> None:
    handle_outputs(
        outputs=[output_metric],
        duration_s=duration_s,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_mode=False,
    )


def _handle_non_stream_request(url, headers, payload) -> None:
    output_metric = OutputMetric(payload=payload)
    start_time = time.perf_counter()
    response = None

    try:
        response = requests.post(url=url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        color_print(
            json.dumps(response_json, indent=2, ensure_ascii=False),
            Color.LIGHT_GREEN,
        )
        output_metric.update_non_stream_response(response_json)
        output_metric.success = True
    except Exception as error:
        if response is not None:
            _print_request_error(response, error)
            output_metric.error_message = (
                f"Status Code={response.status_code}, Reason: {response.text}; {error}"
            )
        else:
            color_print(f"Request Error: {error}", Color.RED)
            output_metric.error_message = str(error)

    duration_s = time.perf_counter() - start_time
    output_metric.latency_ms = duration_s * 1000
    _handle_request_output(output_metric, duration_s)


async def _handle_stream_request(
    url, headers, payload, raw: bool, no_print: bool
) -> None:
    start_time = time.perf_counter()

    async with _create_bench_client_session() as session:
        output_metric = await request_func(
            session,
            request_url=url,
            payload=payload,
            headers=headers,
            raw=raw,
            render_content=not no_print,
        )
    duration_s = time.perf_counter() - start_time
    _handle_request_output(output_metric, duration_s)


def http_request(args):
    url, payload = build_request(args)
    headers = {"Authorization": f"Bearer {args.api_key}"}

    if args.verbose:
        info_print(headers, payload, url)
    if args.disable_stream:
        _handle_non_stream_request(url, headers, payload)
    else:
        asyncio.run(
            _handle_stream_request(url, headers, payload, args.raw, args.no_print)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="aib req", description="Send a single inference request"
    )
    add_common_args(parser)

    parser.add_argument("--dump-path", help="The dump path, jsonl format")
    parser.add_argument(
        "--dump-content",
        default="all",
        choices=["all", "msg"],
        help="The dump Content, jsonl format",
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        help="Optional path to dump the printed metric tables. JSON for write, JSONL for append",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print request details"
    )

    parser.add_argument(
        "--disable-stream", action="store_true", help="Disable streaming"
    )
    parser.add_argument("--prompt", type=str, default="Who are you", help="User prompt")
    parser.add_argument(
        "--output-len",
        type=int,
        help="Target output token length for a random-token request",
    )

    parser.add_argument(
        "--no-print", action="store_true", help="Whether to print sse content"
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
        "--dataset",
        choices=["gsm8k"],
        help="Randomly select one request from a packaged dataset",
    )
    mutex_group.add_argument(
        "--input-len",
        type=int,
        help="Target input token length for a random-token request",
    )
    mutex_group.add_argument(
        "--ebnf", action="store_true", help="Constrained Decoding for EBNF format"
    )
    mutex_group.add_argument(
        "--json-schema-response-format",
        action="store_true",
        help="JSON Schema Response Format",
    )
    mutex_group.add_argument("--tools", action="store_true", help="Add tool")
    mutex_group.add_argument(
        "--payload-path", type=str, help="Path to one JSON request payload"
    )
    mutex_group.add_argument(
        "--input-ids-path", type=str, help="Path to a JSON input token ID array"
    )

    args = parser.parse_args(argv)
    if (args.input_len is None) != (args.output_len is None):
        parser.error("--input-len and --output-len must be used together")
    if args.input_len is not None and args.input_len < 1:
        parser.error("--input-len must be at least 1")
    if args.output_len is not None and args.output_len < 1:
        parser.error("--output-len must be at least 1")
    if args.override_payload is not None:
        try:
            parse_override_payload(args.override_payload)
        except ValueError as error:
            parser.error(str(error))

    try:
        http_request(args)
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
