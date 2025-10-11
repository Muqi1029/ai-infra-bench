import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="mycmd", description="生成客户端或 SGL 模板")
    script_dir = Path(__file__).parent
    src = script_dir.parent / "examples"
    print(f"{src=}")
    parser.add_argument("mode", choices=["sgl", "client"])
    parser.add_argument("command", choices=["gen", "cmp", "slo"], help="bench type")
    parser.add_argument(
        "target", type=Path, nargs="?", default=Path("."), help="the objective path"
    )

    args = parser.parse_args()

    if args.mode == "sgl":
        target = args.target
        if args.command == "gen":
            # TODO: sgl gen 的逻辑
            print(f"SGL模式: gen -> {target}")
            shutil.copy(src / "general_bench.py", target)
        elif args.command == "slo":
            # TODO: sgl slo 的逻辑
            print(f"SGL模式: slo -> {target}")
    elif args.mode == "client":
        if args.command == "gen":
            shutil.copy(src / "client_gen.py", target)
            print(f"已生成 normal/gen: {target}")
        elif args.command == "slo":
            shutil.copy("examples/client_slo.py", target)
            print(f"已生成 normal/slo: {target}")
