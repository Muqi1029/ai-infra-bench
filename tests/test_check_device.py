"""CUDA inspection works without GPU hardware or third-party libraries."""

import ctypes as ct
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from ai_infra_bench import check_device as cd
from ai_infra_bench.cli import main


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


def write(pointer, value):
    pointer._obj.value = value
    return 0


class FakeLibrary:
    def __init__(self, count=2, attribute_error=1):
        values = {
            75: 8,
            76: 9,
            16: 142,
            13: 2_000_000,
            36: 9_000_000,
            37: 384,
            38: 96 * 1024**2,
        }
        self.cuInit = FakeFunction(lambda flags: 0)
        self.cuDriverGetVersion = FakeFunction(lambda ptr: write(ptr, 12040))
        self.cuDeviceGetCount = FakeFunction(lambda ptr: write(ptr, count))
        self.cuDeviceGet = FakeFunction(lambda ptr, index: write(ptr, index))
        self.cuDeviceGetName = FakeFunction(
            lambda buf, size, dev: write_buffer(buf, b"Fake GPU")
        )
        self.cuDeviceGetPCIBusId = FakeFunction(
            lambda buf, size, dev: write_buffer(buf, b"0000:01:00.0")
        )
        self.cuDeviceTotalMem_v2 = FakeFunction(
            lambda ptr, dev: write(ptr, 48 * 1024**3)
        )
        self.cuDeviceGetAttribute = FakeFunction(
            lambda ptr, attr, dev: (
                write(ptr, values[attr]) if attr in values else attribute_error
            )
        )
        self.cuGetErrorString = FakeFunction(
            lambda status, ptr: write(ptr, b"test driver error")
        )


def write_buffer(buffer, value):
    buffer.value = value
    return 0


@pytest.fixture
def driver():
    with patch.object(ct, "CDLL", return_value=FakeLibrary()) as loader:
        yield loader


def test_default_all_and_units(driver, capsys):
    assert main(["check-device", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["visible_device_count"] == 2
    assert report["cuda_driver_api_version"] == "12.4"
    assert [d["index"] for d in report["devices"]] == [0, 1]
    gpu = report["devices"][0]
    assert gpu["total_memory_bytes"] == 48 * 1024**3  # size_t, not 32-bit
    assert gpu["sm_version"] == "sm_89"
    assert gpu["sm_count"] == 142
    assert gpu["l2_cache_bytes"] == 96 * 1024**2
    assert gpu["gpu_clock_period_ns"] == 0.5
    assert gpu["estimated_memory_bandwidth_gb_s"] == 864
    assert gpu["l1_cache_bytes"] is None
    assert gpu["shared_memory_per_block_optin_bytes"] is None


def test_selection_and_text(driver, capsys):
    assert main(["check-device", "--device", "1", "--device", "1"]) == 0
    output = capsys.readouterr().out
    assert output.count("GPU 1:") == 1
    assert "GPU 0:" not in output
    assert "N/A" in output
    assert "2,000.00 MHz" in output
    assert [d["index"] for d in cd.collect_devices([1, 0])["devices"]] == [1, 0]


@pytest.mark.parametrize("index", ["-1", "2"])
def test_invalid_selection(driver, capsys, index):
    assert main(["check-device", "--device", index, "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid device index" in captured.err


def test_no_devices(driver, capsys):
    driver.return_value = FakeLibrary(count=0)
    assert main(["check-device"]) == 1
    assert "No CUDA-visible" in capsys.readouterr().err


def test_missing_driver(driver, capsys):
    driver.side_effect = OSError("missing")
    assert main(["check-device"]) == 1
    assert "Install the NVIDIA driver" in capsys.readouterr().err


def test_unexpected_attribute_error_is_not_hidden(driver):
    driver.return_value = FakeLibrary(attribute_error=999)
    with pytest.raises(cd.DeviceQueryError, match="CUDA error 999"):
        cd.collect_devices()


def test_failed_init(driver, capsys):
    driver.return_value.cuInit = FakeFunction(lambda flags: 100)
    assert main(["check-device"]) == 1
    assert "cuInit:" in capsys.readouterr().err


def test_stdlib_only():
    script = """
import sys
from unittest.mock import patch
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('torch must not be imported')
sys.meta_path.insert(0, BlockTorch())
from ai_infra_bench.cli import main
from ai_infra_bench.check_device import collect_devices
try:
    main(['check-device', '--help'])
except SystemExit as exc:
    assert exc.code == 0
with patch('ctypes.CDLL', side_effect=OSError('missing')):
    assert main(['check-device']) == 1
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def compute(name, cc, sms, clock):
    return cd.estimate_compute(
        {
            "name": name,
            "compute_capability_major": cc[0],
            "compute_capability_minor": cc[1],
            "sm_count": sms,
            "gpu_clock_khz": clock,
        }
    )


@pytest.mark.parametrize(
    "name,cc,sms,clock,cuda,tensor,fp32,fp16",
    [
        ("NVIDIA L40", (8, 9), 142, 2490000, 18176, 568, 90.51648, 181.03296),
        ("NVIDIA A100-SXM4-80GB", (8, 0), 108, 1410000, 6912, 432, 19.49184, 311.86944),
        (
            "NVIDIA H100 80GB HBM3",
            (9, 0),
            132,
            1830000,
            16896,
            528,
            61.83936,
            989.42976,
        ),
        ("Tesla T4", (7, 5), 40, 1590000, 2560, 320, 8.1408, 65.1264),
    ],
)
def test_compute_reference_devices(name, cc, sms, clock, cuda, tensor, fp32, fp16):
    result = compute(name, cc, sms, clock)
    assert result["cuda_core_count"] == cuda
    assert result["tensor_core_count"] == tensor
    assert result["estimated_fp32_tflops"] == pytest.approx(fp32)
    assert result["estimated_tensor_fp16_dense_tflops"] == pytest.approx(fp16)
    if cc[0] >= 8:
        assert result["estimated_tensor_fp16_sparse_tflops"] == pytest.approx(fp16 * 2)
    else:
        assert result["estimated_tensor_fp16_sparse_tflops"] is None


def test_unknown_architecture_and_product():
    unknown = compute("Future GPU", (99, 0), 100, 2000000)
    assert unknown["cuda_core_count"] is None
    assert unknown["tensor_core_count"] is None
    assert unknown["estimated_fp32_tflops"] is None
    # H20 must not inherit H100's throughput simply because both expose sm_90.
    h20 = compute("NVIDIA H20", (9, 0), 78, 1410000)
    assert h20["tensor_core_count"] == 312
    assert h20["estimated_tensor_fp16_dense_tflops"] is None
    assert (
        compute("NVIDIA L40FAKE", (8, 9), 142, 2490000)["tensor_throughput_profile"]
        is None
    )
    assert (
        compute("NVIDIA L40S", (8, 9), 142, 2520000)["tensor_throughput_profile"]
        == "L40S"
    )


def test_no_tensor_cores_and_missing_clock():
    gtx = compute("NVIDIA GeForce GTX 1660 Ti", (7, 5), 24, 1770000)
    assert gtx["tensor_core_count"] == 0
    assert gtx["estimated_tensor_fp16_dense_tflops"] is None
    for clock in (None, 0):
        result = compute("NVIDIA L40", (8, 9), 142, clock)
        assert result["cuda_core_count"] == 18176
        assert result["estimated_fp32_tflops"] is None
        assert result["estimated_tensor_fp16_dense_tflops"] is None
    assert compute("NVIDIA L40", (8, 9), None, 2490000)["cuda_core_count"] is None


def test_tensor_precision_and_visible_sm_scaling():
    full = compute("NVIDIA H100 80GB HBM3", (9, 0), 132, 1830000)
    partition = compute("NVIDIA H100 80GB HBM3 MIG 1g.10gb", (9, 0), 16, 1830000)
    assert partition["tensor_core_count"] == 64
    assert partition["estimated_tensor_fp16_dense_tflops"] == pytest.approx(
        full["estimated_tensor_fp16_dense_tflops"] * 16 / 132
    )
    assert (
        full["estimated_tensor_fp8_dense_tflops"]
        == full["estimated_tensor_int8_dense_tops"]
    )
    assert (
        full["estimated_tensor_tf32_dense_tflops"] * 2
        == full["estimated_tensor_bf16_dense_tflops"]
    )
    assert (
        compute("NVIDIA A100", (8, 0), 108, 1410000)[
            "estimated_tensor_fp8_dense_tflops"
        ]
        is None
    )


def test_l40_and_l40s_have_different_tensor_rates():
    l40 = compute("NVIDIA L40", (8, 9), 142, 2490000)
    l40s = compute("NVIDIA L40S", (8, 9), 142, 2490000)
    assert l40["tensor_core_count"] == l40s["tensor_core_count"] == 568
    assert (
        l40s["estimated_tensor_fp16_dense_tflops"]
        == 2 * l40["estimated_tensor_fp16_dense_tflops"]
    )
    assert l40["estimated_tensor_int8_dense_tops"] == pytest.approx(362.06592)
