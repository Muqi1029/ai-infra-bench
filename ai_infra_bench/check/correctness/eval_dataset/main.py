import asyncio
from argparse import ArgumentParser

from ai_infra_bench.check.correctness.eval_dataset.base import (
    DATASET_CHOICES,
    EvalRuntime,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9090")
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)

    parser.add_argument("--evals", nargs="+", choices=DATASET_CHOICES)

    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--override-payload", type=str)
    return parser.parse_args()


def main():
    runtime_args = parse_args()
    print(f"{runtime_args=}")
    eval_runtime = EvalRuntime(runtime_args)
    asyncio.run(eval_runtime.run())


if __name__ == "__main__":
    main()
