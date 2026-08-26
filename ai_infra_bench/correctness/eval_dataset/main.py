import asyncio
import logging
from argparse import ArgumentParser

from ai_infra_bench.correctness.eval_dataset.base import (
    AGENT_DATASET_CHOICES,
    DATASET_CHOICES,
    DEEPSWE_EVAL,
    EvalRuntime,
)
from ai_infra_bench.utils.req import add_common_args

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = ArgumentParser()
    add_common_args(parser)

    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Concurrent requests or trials (default: 32; DeepSWE: 1)",
    )
    parser.add_argument(
        "--repeat", type=int, help="Evaluation rounds or attempts per task (default: 1)"
    )

    parser.add_argument(
        "--evals",
        nargs="+",
        choices=["all", *DATASET_CHOICES, *AGENT_DATASET_CHOICES],
        required=True,
        help="Evaluations to run; all covers the HTTP datasets, not DeepSWE",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Dataset YAML, or a Pier job config when running DeepSWE",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Dataset path, DeepSWE repository, tasks directory, or single task",
    )
    parser.add_argument(
        "--num-shots", type=int, default=5, help="Few-shot count for HTTP datasets"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        help="Maximum questions, or sampled tasks for DeepSWE",
    )

    deepswe_group = parser.add_argument_group("DeepSWE")
    deepswe_group.add_argument(
        "--deepswe-agent",
        help="Pier agent (default: mini-swe-agent unless --config defines agents)",
    )
    deepswe_group.add_argument(
        "--deepswe-environment",
        choices=["docker", "modal"],
        help="Pier sandbox environment (default: docker)",
    )
    deepswe_group.add_argument(
        "--deepswe-jobs-dir", help="Directory where Pier writes job results"
    )
    deepswe_group.add_argument(
        "--deepswe-sample-seed",
        type=int,
        help="Deterministic task sampling seed (default: 0 with --num-questions)",
    )
    deepswe_group.add_argument(
        "--deepswe-env-file", help="Environment file passed to Pier"
    )
    deepswe_group.add_argument(
        "--deepswe-yes",
        action="store_true",
        help="Auto-confirm Pier host environment access prompts",
    )

    args = parser.parse_args(argv)
    if args.config and len(args.evals) != 1:
        parser.error("--config can only be used with exactly one --evals value")
    if args.num_shots < 0:
        parser.error("--num-shots must be non-negative")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.num_questions is not None and args.num_questions < 1:
        parser.error("--num-questions must be positive")

    if DEEPSWE_EVAL in args.evals:
        if len(args.evals) != 1:
            parser.error("deepswe must be run separately from HTTP dataset evaluations")
        if not args.model and not args.config:
            parser.error("deepswe requires --model or a Pier --config")
        if args.override_payload:
            parser.error("--override-payload is not supported by deepswe")
        if args.max_concurrency is None and not args.config:
            args.max_concurrency = 1
        if args.repeat is None and not args.config:
            args.repeat = 1
        return args

    if "all" in args.evals:
        logger.info(f"--evals has all, means eval all datasets: {DATASET_CHOICES}")
        args.evals = DATASET_CHOICES

    args.base_url = args.base_url or "http://localhost:9090"
    args.max_concurrency = args.max_concurrency or 32
    args.repeat = args.repeat or 1
    if not args.base_url.startswith(("http://", "https://")):
        args.base_url = f"http://{args.base_url}"
    return args


def main(argv=None):
    runtime_args = parse_args(argv)

    if runtime_args.evals == [DEEPSWE_EVAL]:
        from ai_infra_bench.correctness.eval_dataset.deepswe import DeepSWERuntime

        try:
            return DeepSWERuntime(runtime_args).run()
        except (RuntimeError, ValueError) as error:
            logger.error("DeepSWE evaluation failed: %s", error)
            return 1

    eval_runtime = EvalRuntime(runtime_args)
    return asyncio.run(eval_runtime.run())


if __name__ == "__main__":
    raise SystemExit(main())
