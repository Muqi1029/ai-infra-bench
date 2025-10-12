import argparse
import shutil
from importlib import resources
from pathlib import Path

import ai_infra_bench


def main():
    parser = argparse.ArgumentParser(
        prog="ai-bench-cli",
        description="Generate example templates for SGL or client benchmarking.",
    )

    parser.add_argument(
        "mode", choices=["sgl", "client"], help="Select mode: sgl or client."
    )
    parser.add_argument(
        "command", choices=["gen", "cmp", "slo"], help="Benchmark type."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Destination path (default: current dir).",
    )

    args = parser.parse_args()

    example_dir = resources.files(ai_infra_bench) / "examples"
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    def copy_example(filename: str):
        """Copy a single example file from package data to the target directory."""
        src = example_dir / filename
        dst = target / filename
        if not src.exists():
            print(f"❌ Example file not found: {src}")
            return
        shutil.copy(src, dst)
        print(f"✅ Copied to {dst}")

    if args.mode == "sgl":
        print("🔧 SGL mode")
        if args.command == "gen":
            copy_example("gen_bench.py")
        elif args.command == "slo":
            copy_example("slo_bench.py")
        elif args.command == "cmp":
            copy_example("cmp_bench.py")

    elif args.mode == "client":
        print("🔧 Client mode")
        if args.command == "gen":
            copy_example("client_gen.py")
        elif args.command == "slo":
            copy_example("client_slo.py")
        elif args.command == "cmp":
            copy_example("client_cmp.py")


if __name__ == "__main__":
    main()
