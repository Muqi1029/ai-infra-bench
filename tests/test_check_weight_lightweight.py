"""Exercise the CLI in an environment where heavy dependencies cannot import."""

import json
import struct
import subprocess
import sys

import pytest

pytest.importorskip("safetensors")
pytest.importorskip("ml_dtypes")

from ai_infra_bench.check_weight import ParameterStats, load_file, print_weight_summary


def write_checkpoint(path):
    tensors = {
        "model.layers.0.mlp.experts.0.up.weight": ("U8", [2, 3], bytes(6)),
        "model.layers.0.mlp.experts.0.up.weight_scale": (
            "F32",
            [],
            struct.pack("<f", 0.5),
        ),
        "model.layers.0.mlp.experts.0.up.weight_scale_2": ("BF16", [], b"\x80\x3f"),
        "model.embed_tokens.weight": ("BF16", [2, 4], bytes(16)),
        "mtp.norm.weight": ("F8_E4M3", [3], bytes(3)),
        "empty.weight": ("F16", [0, 3], b""),
        "flag": ("BOOL", [], b"\x01"),
    }
    header, data = {}, b""
    for key, (dtype, shape, payload) in tensors.items():
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(payload)],
        }
        data += payload
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return len(tensors), len(data)


def test_metadata_preserves_quantized_counts_and_scalars(tmp_path, capsys):
    path = tmp_path / "model.safetensors"
    expected = write_checkpoint(path)
    stats = ParameterStats()
    assert print_weight_summary(load_file(path), parameter_stats=stats) == expected
    assert stats == ParameterStats(
        total=24, routed_experts=12, embedding_lm_head=8, mtp=3, quant_scales=2
    )
    output = capsys.readouterr().out
    assert "value=0.5" in output
    assert "value=1" in output
    assert "EMPTY tensor" in output
    assert "logical_shape=(2, 6)" in output


def test_local_cli_without_torch_or_hub(tmp_path):
    write_checkpoint(tmp_path / "model.safetensors")
    script = """
import sys
class BlockImports:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'huggingface_hub', 'transformers'}:
            raise ImportError('Blocked heavy/remote dependency: ' + fullname)
sys.meta_path.insert(0, BlockImports())
from ai_infra_bench.cli import main
main(['check-weight', '--path', sys.argv[1]])
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "Checkpoint Summary" in result.stdout
    assert "0.00B (24)" in result.stdout


def test_bin_without_torch_has_actionable_error(tmp_path):
    script = """
import sys
sys.modules['torch'] = None
from ai_infra_bench.check_weight import load_bin_file
try:
    load_bin_file('unused.bin')
except ImportError as exc:
    assert 'use safetensors' in str(exc)
else:
    raise AssertionError('Expected missing torch error')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
