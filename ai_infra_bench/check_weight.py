import argparse
import json
import os
import re
from dataclasses import dataclass
from glob import glob

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from ai_infra_bench.utils.draw import print_table

MIN_NAME_WIDTH = 60
SHAPE_WIDTH = 24
DTYPE_WIDTH = 20
NUMEL_WIDTH = 16
ROUTED_EXPERT_PATTERN = re.compile(r"\.mlp\.experts\.\d+\.")
SHARED_EXPERT_MARKER = ".mlp.shared_experts."


@dataclass
class ParameterStats:
    total: int = 0
    routed_experts: int = 0
    shared_experts: int = 0


def update_parameter_stats(state_dict, stats, items=None):
    items = get_tensor_items(state_dict) if items is None else items
    for key, value in items:
        numel = value.numel() * (
            2 if is_modelopt_nvfp4_weight(key, value, state_dict) else 1
        )
        stats.total += numel
        if ROUTED_EXPERT_PATTERN.search(key):
            stats.routed_experts += numel
        elif SHARED_EXPERT_MARKER in key:
            stats.shared_experts += numel


def activated_parameter_count(stats, config):
    """Estimate parameters used per token by sparse MoE routing."""
    if stats.routed_experts == 0:
        return stats.total

    text_config = config.get("text_config", {})
    if isinstance(text_config, dict):
        config = text_config | config
    n_routed = config.get("n_routed_experts")
    n_shared = config.get("n_shared_experts", 0)
    n_active_routed = config.get("num_experts_per_tok", 1)
    if (
        not isinstance(n_routed, int)
        or n_routed <= 0
        or not isinstance(n_shared, int)
        or n_shared < 0
        or not isinstance(n_active_routed, int)
        or n_active_routed < 0
        or n_active_routed > n_routed
    ):
        return None

    return round(
        stats.total
        - stats.routed_experts
        - stats.shared_experts
        + stats.routed_experts * n_active_routed / n_routed
        + (stats.shared_experts if n_shared else 0)
    )


def format_parameter_count(value):
    if value is None:
        return "N/A"
    return f"{value / 1_000_000_000:.2f}B ({value:,})"


def load_model_config(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def get_tensor_items(state_dict):
    return [
        (key, value)
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    ]


def compile_regex(pattern):
    return re.compile(re.escape(pattern))


def format_shape(value):
    return "scalar (0-D)" if value.ndim == 0 else str(tuple(value.shape))


def format_scalar(value):
    item = value.item()
    return f"{item:.9g}" if isinstance(item, (float, complex)) else str(item)


def is_modelopt_nvfp4_weight(key, value, state_dict):
    """Detect ModelOpt NVFP4 weights without treating every uint8 tensor as FP4."""
    if value.dtype != torch.uint8 or value.ndim == 0 or not key.endswith(".weight"):
        return False

    prefix = key.removesuffix(".weight")
    return (
        f"{prefix}.weight_scale" in state_dict
        and f"{prefix}.weight_scale_2" in state_dict
    )


def tensor_details(key, value, state_dict):
    if value.ndim == 0:
        return f"value={format_scalar(value)}; scalar contains 1 element"

    if value.numel() == 0:
        return "EMPTY tensor: contains 0 elements"

    if is_modelopt_nvfp4_weight(key, value, state_dict):
        logical_shape = list(value.shape)
        logical_shape[-1] *= 2
        return (
            "ModelOpt NVFP4 packed: 2 FP4 values/uint8; "
            f"logical_shape={tuple(logical_shape)}"
        )

    return ""


def print_weight_summary(
    state_dict,
    name_filter=None,
    path=None,
    parameter_stats=None,
):
    items = get_tensor_items(state_dict)
    displayed_items = [
        item for item in items if name_filter is None or name_filter.search(item[0])
    ]
    total_bytes = sum(value.numel() * value.element_size() for _, value in items)
    if parameter_stats is not None:
        update_parameter_stats(state_dict, parameter_stats, items)

    # A name filter is often used across sharded checkpoints. Keep shards with
    # no matching tensors out of the per-file report while retaining their
    # tensors in the overall totals.
    if name_filter is not None and not displayed_items:
        return len(items), total_bytes

    if path is not None:
        print(f"\n🔍 Loading weights from: {path}")

    rows = [
        (key, value, tensor_details(key, value, state_dict))
        for key, value in displayed_items
    ]
    name_width = max(
        MIN_NAME_WIDTH,
        max((len(key) for key, _ in displayed_items), default=len("Layer Name")),
    )
    details_width = max(
        len("Details"),
        max((len(details) for _, _, details in rows), default=0),
    )
    line_width = (
        name_width + SHAPE_WIDTH + DTYPE_WIDTH + NUMEL_WIDTH + details_width + 5
    )

    print("\n📊 Weight Summary:")
    print("-" * line_width)
    print(
        f"{'Layer Name':<{name_width}} "
        f"{'Shape':<{SHAPE_WIDTH}} "
        f"{'Dtype':<{DTYPE_WIDTH}} "
        f"{'Numel':>{NUMEL_WIDTH}}  Details"
    )
    print("-" * line_width)
    for key, value, details in rows:
        print(
            f"{key:<{name_width}} "
            f"{format_shape(value):<{SHAPE_WIDTH}} "
            f"{str(value.dtype):<{DTYPE_WIDTH}} "
            f"{value.numel():>{NUMEL_WIDTH},}  {details}"
        )
    print("-" * line_width)
    print(f"Stored tensor elements: {sum(value.numel() for _, value in items):,}")
    if name_filter is not None:
        print(f"Displayed tensors: {len(displayed_items):,} / {len(items):,}")

    return len(items), total_bytes


def load_bin_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions that do not support weights_only.
        return torch.load(path, map_location="cpu")


def inspect_weight_files(
    weight_files,
    loader,
    name_filter=None,
    parameter_stats=None,
):
    total_tensors = 0
    total_bytes = 0

    for path in weight_files:
        state_dict = loader(path)
        num_tensors, file_bytes = print_weight_summary(
            state_dict,
            name_filter,
            path,
            parameter_stats,
        )
        total_tensors += num_tensors
        total_bytes += file_bytes

    return total_tensors, total_bytes


def download_from_hub(
    model_path,
    cache_dir=None,
    max_checkpoints=-1,
    name_filter=None,
):
    # If cache_dir is None, Hugging Face uses "${HF_HOME}/hub".
    if not os.path.isdir(model_path):
        dir_path = snapshot_download(model_path, cache_dir=cache_dir)
    else:
        dir_path = model_path
    print(f"\n📦 Model downloaded to: {dir_path}\n")

    weight_files = sorted(glob(f"{dir_path}/*.safetensors"))
    loader = load_file

    if not weight_files:
        weight_files = sorted(glob(f"{dir_path}/*.bin"))
        loader = load_bin_file

    if not weight_files:
        raise FileNotFoundError(
            f"No .safetensors or .bin weight files found in {dir_path}"
        )

    if max_checkpoints != -1:
        weight_files = weight_files[:max_checkpoints]

    parameter_stats = ParameterStats()
    total_tensors, total_bytes = inspect_weight_files(
        weight_files,
        loader,
        name_filter,
        parameter_stats,
    )

    config = load_model_config(dir_path)
    summary_rows = [
        ["Metric", "Value"],
        ["Total parameters", format_parameter_count(parameter_stats.total)],
        [
            "Activated parameters",
            format_parameter_count(activated_parameter_count(parameter_stats, config)),
        ],
        ["Num of weight files", f"{len(weight_files):,}"],
        ["Num of tensors", f"{total_tensors:,}"],
        ["Total storage bytes", f"{total_bytes:,}"],
        ["Total storage size", f"{total_bytes / 1024**3:.2f} GiB"],
    ]
    print_table("Weight Summary", summary_rows)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch or safetensors model weights."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=-1)
    parser.add_argument(
        "--name-filter",
        type=compile_regex,
        default=None,
        help="Only display tensor names containing this literal filter string.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    download_from_hub(
        args.model_path,
        cache_dir=args.cache_dir,
        max_checkpoints=args.max_checkpoints,
        name_filter=args.name_filter,
    )


if __name__ == "__main__":
    main()
