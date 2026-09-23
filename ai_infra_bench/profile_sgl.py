"""Start or stop an SGLang server profiler."""

import asyncio
import logging
from argparse import ArgumentParser
from typing import Any, Sequence

from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import add_common_args, api_url

logger = logging.getLogger(__name__)

ACTIVITIES = ("CPU", "GPU", "MEM", "RPD")


def build_payload(args) -> dict[str, Any]:
    """Build a /start_profile body. Optional flags are omitted unless set."""
    payload: dict[str, Any] = {
        "output_dir": args.output_dir,
        "start_step": args.start_step,
        "num_steps": args.num_steps,
        "activities": list(args.activities),
    }
    if args.profile_by_stage:
        payload["profile_by_stage"] = True
    if args.with_stack:
        payload["with_stack"] = True
    if args.record_shapes:
        payload["record_shapes"] = True
    if args.merge_profiles:
        payload["merge_profiles"] = True
    if args.profile_prefix is not None:
        payload["profile_prefix"] = args.profile_prefix
    if args.profile_stages:
        payload["profile_stages"] = list(args.profile_stages)
    return payload


async def send_profile(args) -> int:
    endpoint = "/stop_profile" if args.stop else "/start_profile"
    url = api_url(args.base_url, endpoint)
    headers = {"Authorization": f"Bearer {args.api_key}"}
    payload = None if args.stop else build_payload(args)
    if args.verbose:
        logger.info("POST %s headers=%s payload=%s", url, headers, payload)

    try:
        async with _create_bench_client_session(api_key=args.api_key) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                body = (await response.text()).strip()
                status = response.status
    except Exception:
        logger.exception("Failed to send %s", endpoint)
        return 1

    if status == 200:
        logger.info("%s: %s", endpoint, body or "ok")
        return 0

    logger.error("Request error, status=%s, reason: %s", status, body)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        prog="aib profile-sgl",
        description="Start or stop SGLang profiling",
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-dir",
        default="/tmp/profiles",
        help="Directory where SGLang writes the profile",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=5,
        help="Forward step at which recording starts (inclusive)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of forward steps to record before auto-stop",
    )
    parser.add_argument(
        "--activities",
        nargs="+",
        choices=ACTIVITIES,
        default=["CPU", "GPU"],
        help="Profiler activities to record",
    )
    parser.add_argument(
        "--profile-by-stage",
        action="store_true",
        help="Profile prefill and decode separately",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python stacks for profiled ops",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record operator input shapes",
    )
    parser.add_argument(
        "--merge-profiles",
        action="store_true",
        help="Merge traces from all ranks into one file",
    )
    parser.add_argument(
        "--profile-prefix",
        help="Prefix for the trace filenames",
    )
    parser.add_argument(
        "--profile-stages",
        nargs="+",
        help="Stages to profile when --profile-by-stage is set, e.g. prefill decode",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Call /stop_profile instead of /start_profile",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log the request URL and payload",
    )
    args = parser.parse_args(argv)
    if args.start_step < 0:
        parser.error("--start-step must be >= 0")
    if args.num_steps < 1:
        parser.error("--num-steps must be >= 1")
    return asyncio.run(send_profile(args))


if __name__ == "__main__":
    raise SystemExit(main())
