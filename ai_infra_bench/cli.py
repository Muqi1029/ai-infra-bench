"""Command-line entry point for AI Infra Bench."""

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="aib")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("req", help="Send a simple request", add_help=False)
    subparsers.add_parser("bench", help="Benchmark requests", add_help=False)
    subparsers.add_parser("slo", help="Run a YAML-driven SLO search", add_help=False)
    subparsers.add_parser(
        "reply", help="Reply Payloads for only Input Ids and Output Ids", add_help=False
    )
    subparsers.add_parser(
        "check-weight", help="Check model weights from safetensors", add_help=False
    )
    subparsers.add_parser(
        "plot-metrics", help="Export benchmark metrics to HTML", add_help=False
    )
    subparsers.add_parser(
        "session-bench",
        aliases=["session-reply-bench"],
        help="Replay request sessions with session-level concurrency",
        add_help=False,
    )
    subparsers.add_parser("eval-dataset", help="Evaluate a dataset", add_help=False)
    subparsers.add_parser("eval-logits", help="Evaluate logits", add_help=False)
    subparsers.add_parser(
        "eval-hidden-states", help="Evaluate hidden states", add_help=False
    )
    subparsers.add_parser(
        "monitor", help="Monitor Prometheus targets locally", add_help=False
    )

    args, extra_argv = parser.parse_known_args(argv)
    commands = {
        "req": ("ai_infra_bench.req", "main"),
        "bench": ("ai_infra_bench.performance.bench", "main"),
        "slo": ("ai_infra_bench.slo", "main"),
        "reply": ("ai_infra_bench.reply", "main"),
        "check-weight": ("ai_infra_bench.check_weight", "main"),
        "plot-metrics": ("ai_infra_bench.utils.draw", "main"),
        "session-bench": (
            "ai_infra_bench.performance.session_reply_bench",
            "main",
        ),
        "session-reply-bench": (
            "ai_infra_bench.performance.session_reply_bench",
            "main",
        ),
        "eval-dataset": ("ai_infra_bench.correctness.eval_dataset.main", "main"),
        "eval-logits": ("ai_infra_bench.correctness.logits", "main"),
        "eval-hidden-states": ("ai_infra_bench.correctness.hidden_states", "main"),
        "monitor": ("ai_infra_bench.monitor.cli", "main"),
    }
    module_name, function_name = commands[args.subcommand]
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)(extra_argv)


if __name__ == "__main__":
    raise SystemExit(main())
