from argparse import ArgumentParser
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser(prog="AI Infra Bench(aib)")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser("req", help="Send a simple request", add_help=False)
    subparsers.add_parser("bench", help="Bench request", add_help=False)
    subparsers.add_parser(
        "plot-metrics", help="Export benchmark metrics to HTML", add_help=False
    )
    subparsers.add_parser("eval-dataset", help="Eval Dataset", add_help=False)
    subparsers.add_parser("eval-logits", help="Eval Logits", add_help=False)
    subparsers.add_parser(
        "eval-hidden-states", help="Eval Hidden States", add_help=False
    )
    subparsers.add_parser(
        "monitor", help="Monitor SGLang with Prometheus and Grafana", add_help=False
    )

    args, extra_argv = parser.parse_known_args(argv)

    if args.subcommand == "req":
        from ai_infra_bench.req import main

        return main(extra_argv)
    elif args.subcommand == "bench":
        from ai_infra_bench.performance.bench import main

        return main(extra_argv)
    elif args.subcommand == "plot-metrics":
        from ai_infra_bench.utils.draw import main

        return main(extra_argv)
    elif args.subcommand == "eval-dataset":
        from ai_infra_bench.correctness.eval_dataset.main import main

        return main(extra_argv)

    elif args.subcommand == "eval-logits":
        from ai_infra_bench.correctness.logits import main

        return main(extra_argv)

    elif args.subcommand == "eval-hidden-states":
        from ai_infra_bench.correctness.hidden_states import main

        return main(extra_argv)
    elif args.subcommand == "monitor":
        # from ai_infra_bench.monitoring.cli import main

        # return main(extra_argv)
        pass

    elif args.subcommand == "metrics":
        from ai_infra_bench.monitor import main

        return main(extra_argv)


if __name__ == "__main__":
    raise SystemExit(main())
