import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from ai_infra_bench.check_weight import (
    ParameterStats,
    activated_parameter_count,
    compile_regex,
    download_from_hub,
    format_routed_expert_activation_ratio,
    inspect_weight_files,
    lm_backbone_parameter_count,
    mtp_activated_parameter_count,
    print_weight_summary,
    speculative_module_name,
    update_parameter_stats,
)


def test_name_filter_treats_dots_as_literal_characters():
    name_filter = compile_regex("layers.1")

    assert name_filter.search("model.layers.1.weight")
    assert not name_filter.search("model.layersX1.weight")


def test_inspect_weight_files_skips_unmatched_shards(capsys):
    state_dicts = {
        "first.safetensors": {
            "model.layers.0.weight": torch.zeros(2, 2),
        },
        "second.safetensors": {
            "model.layers.1.weight": torch.zeros(3, 3),
        },
    }

    totals = inspect_weight_files(
        list(state_dicts),
        state_dicts.__getitem__,
        name_filter=re.compile(r"layers\.1"),
    )

    output = capsys.readouterr().out
    assert "first.safetensors" not in output
    assert "Displayed tensors: 0 / 1" not in output
    assert "second.safetensors" in output
    assert "model.layers.1.weight" in output
    assert totals == (2, 13 * torch.zeros(1).element_size())


def test_print_weight_summary_keeps_unfiltered_output(capsys):
    state_dict = {"model.weight": torch.zeros(2, 2)}

    print_weight_summary(state_dict)

    output = capsys.readouterr().out
    assert "Weight Summary" in output
    assert "model.weight" in output
    assert "Stored tensor elements: 4" in output


def test_final_summary_separates_optional_parameter_groups(tmp_path, capsys):
    save_file(
        {
            "model.layers.0.ple.ple_embedding.weight": torch.zeros(2, 5),
            "model.layers.0.input_layernorm.weight": torch.zeros(4),
            "model.embed_tokens.weight": torch.zeros(3),
            "lm_head.weight": torch.zeros(5),
            "mtp.layers.0.input_layernorm.weight": torch.zeros(6),
            "model.visual.blocks.0.weight": torch.zeros(7),
        },
        tmp_path / "model.safetensors",
    )

    download_from_hub(str(tmp_path))

    output = capsys.readouterr().out
    assert "Model Parameters" in output
    assert "MoE Parameters" in output
    assert "Speculative Parameters" in output
    assert "Checkpoint Summary" in output
    assert "Activated parameters (LM backbone)" in output
    assert "Routed expert parameters (LM backbone)" in output
    assert "Routed expert parameters (speculative)" in output
    assert "Routed expert parameters (total)" in output
    assert "Routed expert activation ratio" in output
    assert "Shared expert parameters (LM backbone)" in output
    assert "Shared expert parameters (speculative)" in output
    assert "Shared expert parameters (total)" in output
    assert "N-gram/PLE parameters" in output
    assert "Embedding/LM head parameters" in output
    assert "Activated parameters (when enabled)" in output
    assert "Vision parameters" in output
    assert "0.00B (10)" in output


def test_activated_parameter_count_uses_nested_moe_config():
    stats = ParameterStats(total=16, routed_experts=4, shared_experts=2)
    config = {
        "text_config": {
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
        }
    }

    assert activated_parameter_count(stats, config) == 14


def test_activated_parameter_count_for_dense_model_is_total():
    stats = ParameterStats(total=16)

    assert activated_parameter_count(stats, {}) == 16


def test_activated_parameter_count_excludes_ngram_ple_parameters():
    stats = ParameterStats(
        total=26,
        routed_experts=4,
        shared_experts=2,
        ngram_ple=10,
    )
    config = {"n_routed_experts": 2, "num_experts_per_tok": 1}

    assert activated_parameter_count(stats, config) == 14


def test_main_and_mtp_activation_are_calculated_separately():
    stats = ParameterStats(
        total=100,
        routed_experts=20,
        ngram_ple=10,
        embedding_lm_head=8,
        mtp=30,
        mtp_routed_experts=20,
        mtp_shared_experts=2,
        vision=12,
    )
    config = {"n_routed_experts": 2, "num_experts_per_tok": 1}

    assert activated_parameter_count(stats, config) == 30
    assert lm_backbone_parameter_count(stats) == 40
    assert mtp_activated_parameter_count(stats, config) == 20
    assert format_routed_expert_activation_ratio(stats, config) == "1 / 2 (50.00%)"


def test_speculative_module_name_detects_dspark_from_config():
    stats = ParameterStats(mtp=1)

    assert speculative_module_name(stats, {}) == "MTP"
    assert speculative_module_name(stats, {"dspark_block_size": 5}) == "DSpark"
    assert (
        speculative_module_name(ParameterStats(), {"dspark_block_size": 5})
        == "None detected"
    )


def test_activated_parameter_count_uses_num_experts_and_detected_shared_expert():
    stats = ParameterStats(total=16, routed_experts=4, shared_experts=2)
    config = {
        "text_config": {
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
    }

    assert activated_parameter_count(stats, config) == 14


def test_activated_parameter_count_rejects_invalid_routing_values():
    stats = ParameterStats(total=16, routed_experts=4)

    assert (
        activated_parameter_count(
            stats,
            {"n_routed_experts": 2, "num_experts_per_tok": 3},
        )
        is None
    )


def test_update_parameter_stats_classifies_expert_tensors():
    state_dict = {
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 3),
        "model.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(2),
        "model.layers.0.mlp.shared_experts.up_proj.weight": torch.zeros(2),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(2, 3),
        "layers.0.ffn.shared_experts.w1.weight": torch.zeros(2),
        "model.layers.1.ple.ple_embedding.weight": torch.zeros(2, 5),
        "model.embed_tokens.weight": torch.zeros(3),
        "lm_head.weight": torch.zeros(5),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 3),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(2),
        "mtp.0.ffn.experts.0.w1.weight": torch.zeros(2, 3),
        "mtp.0.ffn.shared_experts.w1.weight": torch.zeros(2),
        "model.visual.blocks.0.attn.proj.weight": torch.zeros(7),
        "model.layers.0.input_layernorm.weight": torch.zeros(4),
    }
    stats = ParameterStats()

    update_parameter_stats(state_dict, stats)

    assert stats == ParameterStats(
        total=63,
        routed_experts=12,
        shared_experts=6,
        ngram_ple=10,
        embedding_lm_head=8,
        mtp=16,
        mtp_routed_experts=12,
        mtp_shared_experts=4,
        vision=7,
    )


def test_update_parameter_stats_expands_packed_nvfp4_weights():
    state_dict = {
        "model.weight": torch.zeros(3, dtype=torch.uint8),
        "model.weight_scale": torch.ones(1),
        "model.weight_scale_2": torch.ones(1),
    }
    stats = ParameterStats()

    update_parameter_stats(state_dict, stats)

    assert stats.total == 8


def test_update_parameter_stats_expands_configured_expert_fp4_weights(capsys):
    state_dict = {
        "mtp.0.ffn.experts.0.w1.weight": torch.zeros(3, dtype=torch.int8),
        "mtp.0.ffn.experts.0.w1.scale": torch.ones(1),
    }
    config = {"expert_dtype": "fp4"}
    stats = ParameterStats()

    update_parameter_stats(state_dict, stats, model_config=config)
    print_weight_summary(state_dict, model_config=config)

    assert stats.total == 7
    assert stats.mtp == 7
    assert stats.mtp_routed_experts == 7
    assert "Packed expert FP4" in capsys.readouterr().out
