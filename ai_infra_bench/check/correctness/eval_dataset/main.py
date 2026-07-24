import asyncio
import logging
from argparse import ArgumentParser

from ai_infra_bench.check.correctness.eval_dataset.base import (
    DATASET_CHOICES,
    EvalRuntime,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9090")
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)

    parser.add_argument(
        "--evals", nargs="+", choices=["all", *DATASET_CHOICES], required=True
    )

    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--override-payload", type=str)
    args = parser.parse_args()
    if args.config and len(args.evals) != 1:
        parser.error("--config can only be used with exactly one --evals value")
    if args.num_shots < 0:
        parser.error("--num-shots must be non-negative")

    if "all" in args.evals:
        logger.info(f"--evals has all, means eval all datasets: {DATASET_CHOICES}")
        args.evals = DATASET_CHOICES
    return args


def main():
    runtime_args = parse_args()
    print(f"{runtime_args=}")
    eval_runtime = EvalRuntime(runtime_args)
    asyncio.run(eval_runtime.run())


if __name__ == "__main__":
    main()
