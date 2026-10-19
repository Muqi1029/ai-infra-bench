"""Command-line entry point for AI Infra Bench."""

import argparse
import importlib
from typing import Any, Callable, NamedTuple, Sequence


class SubcommandSpec(NamedTuple):
    target: str  # format: "module_path:function_name"
    help: str
    aliases: list[str] = []


_COMMANDS: dict[str, SubcommandSpec] = {
    "req": SubcommandSpec(
        "ai_infra_bench.req:main",
        "Send a simple request",
    ),
    "bench": SubcommandSpec(
        "ai_infra_bench.performance.bench:main",
        "Benchmark requests",
    ),
    "slo": SubcommandSpec(
        "ai_infra_bench.slo:main",
        "Run a YAML-driven SLO search",
    ),
    "reply": SubcommandSpec(
        "ai_infra_bench.reply:main",
        "Reply Payloads for only Input Ids and Output Ids",
    ),
    "check-weight": SubcommandSpec(
        "ai_infra_bench.check_weight:main",
        "Check model weights from safetensors",
    ),
    "plot-metrics": SubcommandSpec(
        "ai_infra_bench.utils.draw:main",
        "Export benchmark metrics to HTML",
    ),
    "session-bench": SubcommandSpec(
        "ai_infra_bench.performance.session_reply_bench:main",
        "Replay request sessions with session-level concurrency",
        aliases=["session-reply-bench"],
    ),
    "eval-dataset": SubcommandSpec(
        "ai_infra_bench.correctness.eval_dataset.main:main",
        "Evaluate a dataset",
    ),
    "eval-logits": SubcommandSpec(
        "ai_infra_bench.correctness.logits:main",
        "Evaluate logits",
    ),
    "eval-hidden-states": SubcommandSpec(
        "ai_infra_bench.correctness.hidden_states:main",
        "Evaluate hidden states",
    ),
    "monitor": SubcommandSpec(
        "ai_infra_bench.monitor.cli:main",
        "Monitor Prometheus targets locally",
    ),
    "profile-sgl": SubcommandSpec(
        "ai_infra_bench.profile_sgl:main",
        "Start or stop SGLang profiling",
    ),
    "token": SubcommandSpec(
        "ai_infra_bench.tokenize:main",
        "Encode or decode tokens with a Hugging Face tokenizer",
        aliases=["tokenize"],
    ),
}


def _load_command(target: str) -> Callable[[Sequence[str]], Any]:
    module_name, function_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="aib")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for name, spec in _COMMANDS.items():
        sub = subparsers.add_parser(
            name,
            help=spec.help,
            aliases=spec.aliases,
            add_help=False,
        )
        sub.set_defaults(_target=spec.target)

    args, extra_argv = parser.parse_known_args(argv)

    return _load_command(args._target)(extra_argv)


if __name__ == "__main__":
    raise SystemExit(main())
