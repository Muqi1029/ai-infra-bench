from argparse import Namespace

from ai_infra_bench.utils.req import sanitize_url


def add_common_args(parser: Namespace):
    parser.add_argument(
        "--base-url",
        default="127.0.0.1:8888",
        type=sanitize_url,
        help="The base URL of the router",
    )
    parser.add_argument(
        "--api-key", default="JustKeepMe", help="The API key of the router"
    )
    parser.add_argument("--model", type=str, help="The model to benchmark")

    parser.add_argument("--override-payload", type=str, help="Override the payload")

    parser.add_argument("--seed", type=int, default=42, help="The seed for random")

    parser.add_argument(
        "--max-concurrency", default=32, type=int, help="The max concurrency"
    )
    parser.add_argument(
        "--request-rate", default=float("inf"), type=float, help="Request rate"
    )
    parser.add_argument(
        "--payload-regex-path", type=str, help="The path of requests", required=True
    )

    parser.add_argument(
        "--completion-tokens-output-path",
        type=str,
        default=None,
        help="Optional path to dump the full completion_tokens list",
    )
    parser.add_argument(
        "--finish-reason-length-output-path",
        type=str,
        default=None,
        help="Optional path to dump outputs whose finish_reason is 'length'",
    )

    parser.add_argument("--label", help="Label used for discribe this benchmark")

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

    parser.add_argument("--debug", action="store_true", help="Debug mode")
