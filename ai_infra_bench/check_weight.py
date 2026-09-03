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
ROUTED_EXPERT_PATTERN = re.compile(r"\.(?:mlp|ffn)\.experts\.\d+\.")
SHARED_EXPERT_PATTERN = re.compile(r"\.(?:mlp|ffn)\.shared_experts?\.")
NGRAM_PLE_PATTERN = re.compile(r"(?:^|\.)ple\.")
MTP_PATTERN = re.compile(r"(?:^|\.)mtp\.")
VISION_PATTERN = re.compile(r"(?:^|\.)(?:visual|vision_model|vision_tower)\.")
EMBEDDING_LM_HEAD_PATTERN = re.compile(
    r"(?:^|\.)(?:embed|embed_tokens|word_embeddings|wte|head|lm_head)\."
)


@dataclass
class ParameterStats:
    total: int = 0
    routed_experts: int = 0
    shared_experts: int = 0
    ngram_ple: int = 0
    embedding_lm_head: int = 0
    mtp: int = 0
    mtp_routed_experts: int = 0
    mtp_shared_experts: int = 0
    vision: int = 0


def update_parameter_stats(state_dict, stats, items=None, model_config=None):
    items = get_tensor_items(state_dict) if items is None else items
    for key, value in items:
        numel = value.numel() * (
            2 if is_packed_fp4_weight(key, value, state_dict, model_config) else 1
        )
        stats.total += numel
        if NGRAM_PLE_PATTERN.search(key):
            stats.ngram_ple += numel
        elif MTP_PATTERN.search(key):
            stats.mtp += numel
            if ROUTED_EXPERT_PATTERN.search(key):
                stats.mtp_routed_experts += numel
            elif SHARED_EXPERT_PATTERN.search(key):
                stats.mtp_shared_experts += numel
        elif VISION_PATTERN.search(key):
            stats.vision += numel
        elif EMBEDDING_LM_HEAD_PATTERN.search(key):
            stats.embedding_lm_head += numel
        elif ROUTED_EXPERT_PATTERN.search(key):
            stats.routed_experts += numel
        elif SHARED_EXPERT_PATTERN.search(key):
            stats.shared_experts += numel


def moe_routing(config):
    """Return the configured active and total routed-expert counts."""
    text_config = config.get("text_config", {})
    if isinstance(text_config, dict):
        config = text_config | config
    n_routed = config.get("n_routed_experts")
    if n_routed is None:
        n_routed = config.get("num_experts")
    n_active_routed = config.get("num_experts_per_tok", 1)
    if (
        not isinstance(n_routed, int)
        or n_routed <= 0
        or not isinstance(n_active_routed, int)
        or n_active_routed < 0
        or n_active_routed > n_routed
    ):
        return None

    return n_active_routed, n_routed


def moe_activated_parameter_count(total, routed_experts, config):
    """Apply the configured MoE routing fraction to one parameter group."""
    if routed_experts == 0:
        return total

    routing = moe_routing(config)
    if routing is None:
        return None
    n_active_routed, n_routed = routing

    return round(total - routed_experts + routed_experts * n_active_routed / n_routed)


def format_routed_expert_activation_ratio(stats, config):
    routed_experts = stats.routed_experts + stats.mtp_routed_experts
    if routed_experts == 0:
        return "N/A"

    routing = moe_routing(config)
    if routing is None:
        return "N/A"
    n_active_routed, n_routed = routing
    return f"{n_active_routed} / {n_routed} ({n_active_routed / n_routed:.2%})"


def activated_parameter_count(stats, config):
    """Estimate activated parameters in the main language-model backbone.

    N-gram/PLE tables, token embeddings, the output head, MTP, and vision are
    reported separately. Shared experts remain part of the backbone total;
    routed experts are reduced to the configured per-token routing fraction.
    """
    backbone_total = lm_backbone_parameter_count(stats)
    return moe_activated_parameter_count(
        backbone_total,
        stats.routed_experts,
        config,
    )


def lm_backbone_parameter_count(stats):
    return (
        stats.total
        - stats.ngram_ple
        - stats.embedding_lm_head
        - stats.mtp
        - stats.vision
    )


def mtp_activated_parameter_count(stats, config):
    """Estimate activation when optional MTP or DSpark weights are used.

    DeepSeek DSpark stages use the ``mtp.<stage>.*`` checkpoint namespace, so
    they are covered by the same weight-derived accounting as regular MTP.
    """
    return moe_activated_parameter_count(
        stats.mtp,
        stats.mtp_routed_experts,
        config,
    )


def speculative_module_name(stats, config):
    if stats.mtp == 0:
        return "None detected"

    text_config = config.get("text_config", {})
    configs = [config, text_config] if isinstance(text_config, dict) else [config]
    if any(
        cfg.get("dspark_block_size") or cfg.get("dspark_target_layer_ids")
        for cfg in configs
    ):
        return "DSpark"
    return "MTP"


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


def is_configured_expert_fp4_weight(key, value, model_config):
    """Detect packed expert FP4 weights declared by the model config."""
    if not isinstance(model_config, dict):
        return False

    text_config = model_config.get("text_config", {})
    if isinstance(text_config, dict):
        model_config = text_config | model_config
    expert_dtype = model_config.get("expert_dtype")
    return (
        isinstance(expert_dtype, str)
        and expert_dtype.lower() == "fp4"
        and value.dtype in (torch.int8, torch.uint8)
        and value.ndim > 0
        and key.endswith(".weight")
        and ROUTED_EXPERT_PATTERN.search(key) is not None
    )


def is_packed_fp4_weight(key, value, state_dict, model_config=None):
    return is_modelopt_nvfp4_weight(
        key, value, state_dict
    ) or is_configured_expert_fp4_weight(key, value, model_config)


def tensor_details(key, value, state_dict, model_config=None):
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

    if is_configured_expert_fp4_weight(key, value, model_config):
        logical_shape = list(value.shape)
        logical_shape[-1] *= 2
        return (
            "Packed expert FP4: 2 FP4 values/int8; "
            f"logical_shape={tuple(logical_shape)}"
        )

    return ""


def print_weight_summary(
    state_dict,
    name_filter=None,
    path=None,
    parameter_stats=None,
    model_config=None,
):
    items = get_tensor_items(state_dict)
    displayed_items = [
        item for item in items if name_filter is None or name_filter.search(item[0])
    ]
    total_bytes = sum(value.numel() * value.element_size() for _, value in items)
    if parameter_stats is not None:
        update_parameter_stats(state_dict, parameter_stats, items, model_config)

    # A name filter is often used across sharded checkpoints. Keep shards with
    # no matching tensors out of the per-file report while retaining their
    # tensors in the overall totals.
    if name_filter is not None and not displayed_items:
        return len(items), total_bytes

    if path is not None:
        print(f"\n🔍 Loading weights from: {path}")

    rows = [
        (key, value, tensor_details(key, value, state_dict, model_config))
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
    model_config=None,
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
            model_config,
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

    config = load_model_config(dir_path)
    parameter_stats = ParameterStats()
    total_tensors, total_bytes = inspect_weight_files(
        weight_files,
        loader,
        name_filter,
        parameter_stats,
        config,
    )

    model_rows = [
        ["Metric", "Value"],
        ["Total parameters", format_parameter_count(parameter_stats.total)],
        [
            "LM backbone parameters",
            format_parameter_count(lm_backbone_parameter_count(parameter_stats)),
        ],
        [
            "Activated parameters (LM backbone)",
            format_parameter_count(activated_parameter_count(parameter_stats, config)),
        ],
        [
            "N-gram/PLE parameters",
            format_parameter_count(parameter_stats.ngram_ple),
        ],
        [
            "Embedding/LM head parameters",
            format_parameter_count(parameter_stats.embedding_lm_head),
        ],
        ["Vision parameters", format_parameter_count(parameter_stats.vision)],
    ]
    moe_rows = [
        ["Metric", "Value"],
        [
            "Routed expert parameters (LM backbone)",
            format_parameter_count(parameter_stats.routed_experts),
        ],
        [
            "Routed expert parameters (speculative)",
            format_parameter_count(parameter_stats.mtp_routed_experts),
        ],
        [
            "Routed expert parameters (total)",
            format_parameter_count(
                parameter_stats.routed_experts + parameter_stats.mtp_routed_experts
            ),
        ],
        [
            "Routed expert activation ratio",
            format_routed_expert_activation_ratio(parameter_stats, config),
        ],
        [
            "Shared expert parameters (LM backbone)",
            format_parameter_count(parameter_stats.shared_experts),
        ],
        [
            "Shared expert parameters (speculative)",
            format_parameter_count(parameter_stats.mtp_shared_experts),
        ],
        [
            "Shared expert parameters (total)",
            format_parameter_count(
                parameter_stats.shared_experts + parameter_stats.mtp_shared_experts
            ),
        ],
    ]
    speculative_rows = [
        ["Metric", "Value"],
        ["Module", speculative_module_name(parameter_stats, config)],
        [
            "Parameters",
            format_parameter_count(parameter_stats.mtp),
        ],
        [
            "Activated parameters (when enabled)",
            format_parameter_count(
                mtp_activated_parameter_count(parameter_stats, config)
            ),
        ],
    ]
    checkpoint_rows = [
        ["Metric", "Value"],
        ["Num of weight files", f"{len(weight_files):,}"],
        ["Num of tensors", f"{total_tensors:,}"],
        ["Total storage bytes", f"{total_bytes:,}"],
        ["Total storage size", f"{total_bytes / 1024**3:.2f} GiB"],
    ]
    print_table("Model Parameters", model_rows)
    print_table("MoE Parameters", moe_rows)
    print_table("Speculative Parameters", speculative_rows)
    print_table("Checkpoint Summary", checkpoint_rows)


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
