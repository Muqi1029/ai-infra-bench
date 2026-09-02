import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

from ai_infra_bench.check_weight import inspect_weight_files, print_weight_summary


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
