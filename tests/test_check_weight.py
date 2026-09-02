import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

from ai_infra_bench.check_weight import (
    ParameterStats,
    activated_parameter_count,
    compile_regex,
    inspect_weight_files,
    print_weight_summary,
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
        "model.layers.0.mlp.shared_experts.up_proj.weight": torch.zeros(2),
        "model.layers.0.input_layernorm.weight": torch.zeros(4),
    }
    stats = ParameterStats()

    update_parameter_stats(state_dict, stats)

    assert stats == ParameterStats(total=12, routed_experts=6, shared_experts=2)


def test_update_parameter_stats_expands_packed_nvfp4_weights():
    state_dict = {
        "model.weight": torch.zeros(3, dtype=torch.uint8),
        "model.weight_scale": torch.ones(1),
        "model.weight_scale_2": torch.ones(1),
    }
    stats = ParameterStats()

    update_parameter_stats(state_dict, stats)

    assert stats.total == 8
