import argparse
import json
import random
import sys
import time
from typing import Any, Dict, Mapping, Tuple

from ai_infra_bench.utils.color import Color, color_print
from ai_infra_bench.utils.io import _read_json, read_packaged_requests
from ai_infra_bench.utils.req import (
    add_common_args,
    api_url,
    parse_override_payload,
    prepare_payload,
)

RANDOM_TOKEN_UPPER_BOUND = 10_000
response_format_json_schema = {
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


def _read_stdin_payload() -> Dict[str, Any]:
    payload = json.load(sys.stdin)
    if not isinstance(payload, Mapping):
        raise ValueError("stdin must contain a JSON object")
    return payload


def build_request(args) -> Tuple[str, Dict[str, Any]]:
    url = api_url(args.base_url, "/v1/chat/completions")

    if args.dataset:
        payload = _select_dataset_request(args.dataset, args.seed)
    elif args.payload_path == "-":
        payload = _read_stdin_payload()
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
    else:
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        payload = {"messages": [{"role": "user", "content": prompt}]}
        if args.ebnf:
            payload["ebnf"] = ebnf_content
        elif args.response_format_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format_json_schema,
            }
        elif args.tools:
            payload["tools"] = tools
            payload["tool_choice"] = "required"

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

    if args.output_len is not None:
        payload["max_tokens"] = args.output_len
    prepared = prepare_payload(
        payload, args.model, args.override_payload, stream=not args.disable_stream
    )
    if "prompt" in prepared and "messages" not in prepared:
        url = api_url(args.base_url, "/v1/completions")
    return url, prepared


def handle_outputs(*args, **kwargs):
    from ai_infra_bench.performance.bench_utils import handle_outputs as _handle_outputs

    return _handle_outputs(*args, **kwargs)


def _print_request_error(
    status_code: int, text: str, error: Exception | None = None
) -> None:
    suffix = f" Error: {error}" if error is not None else ""
    color_print(
        f"Request Error, Status Code={status_code}, Reason: {text}{suffix}",
        Color.RED,
    )


def _handle_request_output(output_metric, duration_s: float) -> None:
    handle_outputs(
        outputs=[output_metric],
        duration_s=duration_s,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_mode=False,
    )


async def _handle_non_stream_request(url, headers, payload):
    from ai_infra_bench.performance.struct import OutputMetric
    from ai_infra_bench.utils.client import _create_bench_client_session

    output_metric = OutputMetric(payload=payload)
    start_time = time.perf_counter()

    try:
        async with _create_bench_client_session(
            max_concurrency=1, api_key=None
        ) as session:
            async with session.post(url=url, headers=headers, json=payload) as response:
                raw_body = await response.text()
                if response.status != 200:
                    _print_request_error(response.status, raw_body)
                    output_metric.error_message = (
                        f"Status Code={response.status}, Reason: {raw_body}"
                    )
                else:
                    response_json = json.loads(raw_body)
                    if isinstance(response_json, Mapping) and response_json.get(
                        "error"
                    ):
                        err = response_json["error"]
                        detail = (
                            err.get("message") or json.dumps(err, ensure_ascii=False)
                            if isinstance(err, Mapping)
                            else str(err)
                        )
                        color_print(f"Request Error: {detail}", Color.RED)
                        output_metric.error_message = detail
                    else:
                        color_print(
                            json.dumps(response_json, indent=2, ensure_ascii=False),
                            Color.LIGHT_GREEN,
                        )
                        output_metric.update_non_stream_response(response_json)
                        output_metric.success = True
    except Exception as error:
        color_print(f"Request Error: {error}", Color.RED)
        output_metric.error_message = str(error)

    duration_s = time.perf_counter() - start_time
    output_metric.latency_ms = duration_s * 1000
    _handle_request_output(output_metric, duration_s)
    return output_metric


async def _handle_stream_request(url, headers, payload, raw: bool, no_print: bool):
    from ai_infra_bench.performance.core import request_func
    from ai_infra_bench.utils.client import _create_bench_client_session

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
    return output_metric


def http_request(args) -> int:
    import asyncio

    url, payload = build_request(args)
    headers = {"Authorization": f"Bearer {args.api_key}"}

    if args.verbose:
        info_print(headers, payload, url)
    if args.disable_stream:
        output = asyncio.run(_handle_non_stream_request(url, headers, payload))
    else:
        output = asyncio.run(
            _handle_stream_request(url, headers, payload, args.raw, args.no_print)
        )
    return 0 if output.success else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="aib req",
        description=(
            "Send a single inference request. With no arguments, read a JSON "
            "payload from piped or redirected stdin; in a terminal, send the "
            "default prompt 'Who are you'."
        ),
    )
    add_common_args(parser)

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print request details"
    )

    parser.add_argument(
        "--disable-stream", action="store_true", help="Disable streaming"
    )
    parser.add_argument(
        "--prompt",
        "--user-prompt",
        type=str,
        help="User prompt, or '-' to read plain text from stdin",
    )
    parser.add_argument(
        "--output-len",
        "--max-tokens",
        type=int,
        help="Maximum output tokens; required with --input-len",
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
        "--raw", action="store_true", help="Whether to print raw sse content"
    )

    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--dataset",
        choices=["gsm8k", "gpqa"],
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
        "--response-format-json-schema",
        action="store_true",
        help="JSON Schema Response Format",
    )
    mutex_group.add_argument("--tools", action="store_true", help="Add tool")
    mutex_group.add_argument(
        "--payload-path",
        type=str,
        help="Path to one JSON request payload, or '-' to read from stdin",
    )

    args = parser.parse_args(argv)
    if not argv and not sys.stdin.isatty():
        args.payload_path = "-"
    if args.prompt is not None and any(
        (
            args.dataset,
            args.payload_path,
            args.input_len is not None,
        )
    ):
        parser.error("--prompt cannot be combined with another request input")
    if args.prompt is None:
        args.prompt = "Who are you"
    if args.input_len is not None and args.output_len is None:
        parser.error("--input-len requires --output-len or --max-tokens")
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
        return http_request(args)
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
