from argparse import ArgumentParser
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser(prog="AI Infra Bench(aib)")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("req", help="Send a simple request", add_help=False)
    subparsers.add_parser("eval-dataset", help="Eval Dataset", add_help=False)

    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "req":
        from ai_infra_bench.req import main

        main(extra_argv)
    elif args.subcommand == "eval-dataset":
        from ai_infra_bench.correctness.eval_dataset.main import main

        main(extra_argv)

    elif args.subcommand == "eval-logits":
        pass

    elif args.subcommand == "eval-hidden-states":
        pass


if __name__ == "__main__":
    main()
