"""Inspect NVIDIA GPUs using only the standard library and the CUDA driver."""

import argparse
import ctypes as ct
import json
import os
import re
import sys
import textwrap
from typing import Sequence

# CUdevice_attribute values from cuda.h. Query separately so older drivers can
# omit newer attributes without losing the rest of the device report.
_ATTRIBUTES = {
    "max_threads_per_block": 1,
    "shared_memory_per_block_bytes": 8,
    "warp_size": 10,
    "max_registers_per_block": 12,
    "gpu_clock_khz": 13,
    "sm_count": 16,
    "memory_clock_khz": 36,
    "memory_bus_width_bits": 37,
    "l2_cache_bytes": 38,
    "max_threads_per_sm": 39,
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "global_l1_cache_supported": 79,
    "local_l1_cache_supported": 80,
    "shared_memory_per_sm_bytes": 81,
    "max_registers_per_sm": 82,
    "shared_memory_per_block_optin_bytes": 97,
    "max_blocks_per_sm": 106,
    "reserved_shared_memory_per_block_bytes": 111,
}
# FP32 lanes (marketed as CUDA cores) and Tensor Cores per SM.
# NVIDIA CUDA samples Common/helper_cuda.h and architecture whitepapers:
# https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
# https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
# https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
# Do not extrapolate an unknown SM version to the nearest known generation.
_SM_CORES = {
    (5, 0): (128, 0),
    (5, 2): (128, 0),
    (5, 3): (128, 0),
    (6, 0): (64, 0),
    (6, 1): (128, 0),
    (6, 2): (128, 0),
    (7, 0): (64, 8),
    (7, 2): (64, 8),
    (7, 5): (64, 8),
    (8, 0): (64, 4),
    (8, 6): (128, 4),
    (8, 7): (128, 4),
    (8, 9): (128, 4),
    (9, 0): (128, 4),
    (10, 0): (128, 4),
    (12, 0): (128, 4),
}

# Dense operations / SM / cycle, FMA = 2 operations, FP32 accumulation
# for floating-point Tensor modes and INT32 accumulation for INT8.
# Product matching is deliberate: SM version alone does not capture SKU limits
# (e.g. H20 versus H100, or GeForce FP32-accumulation limits).
# Sources: NVIDIA product briefs/specifications, including
# https://www.nvidia.com/en-us/data-center/l40s/
# https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
# (Appendix C: L40 has half the Tensor throughput per SM of L40S.)
# https://www.nvidia.com/en-us/data-center/l4/
# https://www.nvidia.com/en-us/data-center/a100/
# https://www.nvidia.com/en-us/data-center/h100/
_TENSOR_PROFILES = [
    ((7, 0), r"V100", {"fp16": 1024}, False),
    ((7, 5), r"T4", {"fp16": 1024, "int8": 2048}, False),
    (
        (8, 0),
        r"A100|A800|A30",
        {"tf32": 1024, "fp16": 2048, "bf16": 2048, "int8": 4096},
        True,
    ),
    ((8, 6), r"A10|A40", {"tf32": 512, "fp16": 1024, "bf16": 1024, "int8": 2048}, True),
    (
        (8, 9),
        r"L40",
        {"tf32": 256, "fp16": 512, "bf16": 512, "fp8": 1024, "int8": 1024},
        True,
    ),
    (
        (8, 9),
        r"L4|L40S",
        {"tf32": 512, "fp16": 1024, "bf16": 1024, "fp8": 2048, "int8": 2048},
        True,
    ),
    (
        (9, 0),
        r"H100|H200|H800",
        {"tf32": 2048, "fp16": 4096, "bf16": 4096, "fp8": 8192, "int8": 8192},
        True,
    ),
]
_TENSOR_MODES = ("tf32", "fp16", "bf16", "fp8", "int8")


def estimate_compute(device):
    """Derive counts and theoretical peaks without executing any GPU workload."""
    capability = (
        device["compute_capability_major"],
        device["compute_capability_minor"],
    )
    cuda, tensor = _SM_CORES.get(capability, (None, None))
    # TU116/TU117 expose sm_75 but have no Tensor Cores.
    if capability == (7, 5):
        name = device["name"]
        if re.search(r"\bGTX\s+16\d{2}\b", name, re.I):
            tensor = 0
        elif not re.search(r"\b(T4|TITAN RTX|RTX\s+\d+)\b", name, re.I):
            tensor = None
    sms = device["sm_count"]
    clock = device["gpu_clock_khz"]
    scale = sms * clock / 1e9 if sms and clock and sms > 0 and clock > 0 else None
    result = {
        "cuda_cores_per_sm": cuda,
        "tensor_cores_per_sm": tensor,
        "cuda_core_count": cuda * sms if cuda is not None and sms is not None else None,
        "tensor_core_count": (
            tensor * sms if tensor is not None and sms is not None else None
        ),
        "estimated_fp32_tflops": (
            cuda * 2 * scale if cuda is not None and scale is not None else None
        ),
        "tensor_throughput_profile": None,
        "tensor_dense_ops_per_sm_per_cycle": {},
    }
    ops, sparse = {}, False
    for cc, pattern, rates, supports_sparse in _TENSOR_PROFILES:
        match = re.search(r"\b(?:" + pattern + r")\b", device["name"], re.I)
        if cc == capability and match:
            result["tensor_throughput_profile"] = match.group().upper()
            ops, sparse = rates, supports_sparse
            break
    result["tensor_dense_ops_per_sm_per_cycle"] = dict(ops)
    for mode in _TENSOR_MODES:
        unit = "tops" if mode == "int8" else "tflops"
        dense = ops[mode] * scale if mode in ops and scale is not None else None
        result[f"estimated_tensor_{mode}_dense_{unit}"] = dense
        result[f"estimated_tensor_{mode}_sparse_{unit}"] = (
            dense * 2 if dense is not None and sparse else None
        )
    return result


_NOTES = [
    "Device indices are CUDA ordinals and respect CUDA_VISIBLE_DEVICES; they may differ from nvidia-smi indices.",
    "L1 capacity is not exposed by a portable CUDA driver attribute; null means unavailable, not zero.",
    "Clock rates are driver-reported device attributes, not live clock measurements. Period = 1e6 / clock_khz (ns).",
    "Memory bandwidth is a DDR estimate: 2 * memory_clock_khz * 1000 * bus_width_bits / 8 / 1e9 GB/s; not measured bandwidth. The DDR assumption may not match every memory technology.",
    "Core counts are architecture-derived estimates (CUDA cores = FP32 lanes), not driver queries; unknown architectures return N/A.",
    "Compute peaks are estimates, not benchmarks: SM count * operations/SM/cycle * clock_khz / 1e9. FMA counts as 2 operations. Actual clocks, SKU limits, and workload efficiency affect throughput.",
    "Tensor floating-point modes use FP32 accumulation; INT8 uses INT32 accumulation. Sparse peaks require supported 2:4 structured sparsity and are 2x dense. N/A means unknown or unsupported, not zero.",
    "Tensor throughput uses verified product profiles; an unlisted GPU can have Tensor Cores while its throughput remains N/A.",
    "Opt-in shared memory is the per-block limit requiring explicit kernel opt-in; it is not added to the default limit.",
]


class DeviceQueryError(RuntimeError):
    """CUDA driver is unavailable or a device query failed."""


class CudaDriver:
    def __init__(self):
        name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
        loader = ct.WinDLL if os.name == "nt" else ct.CDLL
        try:
            self.lib = loader(name)
        except OSError as exc:
            raise DeviceQueryError(
                f"Cannot load {name}. Install the NVIDIA driver and expose GPU devices "
                "and driver libraries to this environment; CUDA Toolkit and torch are not required."
            ) from exc
        signatures = {
            "cuInit": [ct.c_uint],
            "cuDriverGetVersion": [ct.POINTER(ct.c_int)],
            "cuDeviceGetCount": [ct.POINTER(ct.c_int)],
            "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
            "cuDeviceGetName": [ct.POINTER(ct.c_char), ct.c_int, ct.c_int],
            "cuDeviceGetPCIBusId": [ct.POINTER(ct.c_char), ct.c_int, ct.c_int],
            "cuDeviceTotalMem_v2": [ct.POINTER(ct.c_size_t), ct.c_int],
            "cuDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
        }
        for symbol, argtypes in signatures.items():
            try:
                function = getattr(self.lib, symbol)
            except AttributeError as exc:
                raise DeviceQueryError(
                    f"NVIDIA driver is missing {symbol}; update the driver."
                ) from exc
            function.argtypes = argtypes
            function.restype = ct.c_int
        self.check(self.lib.cuInit(0), "cuInit")

    def check(self, status, operation):
        if status == 0:
            return
        description = ct.c_char_p()
        self.lib.cuGetErrorString(status, ct.byref(description))
        detail = (
            description.value.decode(errors="replace")
            if description.value
            else "unknown error"
        )
        raise DeviceQueryError(f"{operation}: {detail} (CUDA error {status})")

    def integer(self, symbol):
        value = ct.c_int()
        self.check(getattr(self.lib, symbol)(ct.byref(value)), symbol)
        return value.value

    def device(self, index):
        device = ct.c_int()
        self.check(self.lib.cuDeviceGet(ct.byref(device), index), "cuDeviceGet")
        result = {"index": index}
        for key, symbol in (
            ("name", "cuDeviceGetName"),
            ("pci_bus_id", "cuDeviceGetPCIBusId"),
        ):
            buffer = ct.create_string_buffer(256)
            self.check(getattr(self.lib, symbol)(buffer, len(buffer), device), symbol)
            result[key] = buffer.value.decode(errors="replace")
        memory = ct.c_size_t()
        self.check(
            self.lib.cuDeviceTotalMem_v2(ct.byref(memory), device),
            "cuDeviceTotalMem_v2",
        )
        result["total_memory_bytes"] = memory.value
        for key, attribute in _ATTRIBUTES.items():
            value = ct.c_int()
            status = self.lib.cuDeviceGetAttribute(ct.byref(value), attribute, device)
            if status in (1, 801):  # INVALID_VALUE (unknown attribute), NOT_SUPPORTED
                result[key] = None
            else:
                self.check(status, f"cuDeviceGetAttribute({key})")
                result[key] = value.value
        major, minor = (
            result["compute_capability_major"],
            result["compute_capability_minor"],
        )
        result["sm_version"] = (
            f"sm_{major}{minor}" if major is not None and minor is not None else None
        )
        result["l1_cache_bytes"] = None
        for kind in ("gpu", "memory"):
            rate = result[f"{kind}_clock_khz"]
            result[f"{kind}_clock_period_ns"] = 1e6 / rate if rate else None
        rate, width = result["memory_clock_khz"], result["memory_bus_width_bits"]
        result["estimated_memory_bandwidth_gb_s"] = (
            2 * rate * 1000 * width / 8 / 1e9 if rate and width else None
        )
        result.update(estimate_compute(result))
        return result


def collect_devices(indices=None):
    driver = CudaDriver()
    count = driver.integer("cuDeviceGetCount")
    if count == 0:
        raise DeviceQueryError(
            "No CUDA-visible NVIDIA GPUs found. Check the driver, container GPU access, and CUDA_VISIBLE_DEVICES."
        )
    selected = list(range(count)) if indices is None else list(dict.fromkeys(indices))
    for index in selected:
        if not 0 <= index < count:
            raise DeviceQueryError(
                f"Invalid device index {index}; visible CUDA indices are 0..{count - 1}."
            )
    version = driver.integer("cuDriverGetVersion")
    return {
        "cuda_driver_api_version": f"{version // 1000}.{version % 1000 // 10}",
        "visible_device_count": count,
        "devices": [driver.device(index) for index in selected],
        "notes": _NOTES,
    }


def _format_value(key, value):
    if value is None:
        return "N/A"
    if key == "total_memory_bytes":
        return f"{value / 1e9:,.2f} GB ({value / 1024**3:,.2f} GiB)"
    if key.endswith("_bytes"):
        for unit, divisor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
            if value >= divisor:
                return f"{value / divisor:,.2f} {unit}"
        return f"{value:,} B"
    if key.endswith("_supported"):
        return "Yes" if value else "No"
    if key.endswith("_khz"):
        return f"{value / 1000:,.2f} MHz"
    if key.endswith("_ns"):
        return f"{value:,.4f} ns"
    if key.endswith("_gb_s"):
        return f"{value:,.2f} GB/s"
    if key.endswith("_tflops"):
        return f"{value:,.2f} TFLOPS"
    if key.endswith("_tops"):
        return f"{value:,.2f} TOPS"
    if key.endswith("_bits"):
        return f"{value:,} bit"
    return f"{value:,}" if isinstance(value, int) else str(value)


def print_report(report):
    from ai_infra_bench.utils.color import Color, colorize, use_color

    def heading(text):
        return colorize(text, Color.LIGHT_CYAN) if use_color(sys.stdout) else text

    groups = [
        (
            "Architecture",
            [
                ("sm_version", "SM version"),
                ("sm_count", "SM count"),
                ("cuda_cores_per_sm", "CUDA Cores / SM (estimated)"),
                ("cuda_core_count", "CUDA Cores total (estimated)"),
                ("tensor_cores_per_sm", "Tensor Cores / SM (estimated)"),
                ("tensor_core_count", "Tensor Cores total (estimated)"),
                ("warp_size", "Threads / warp"),
                ("max_threads_per_block", "Max threads / block"),
                ("max_threads_per_sm", "Max threads / SM"),
                ("max_blocks_per_sm", "Max blocks / SM"),
                ("max_registers_per_block", "Max registers / block"),
                ("max_registers_per_sm", "Max registers / SM"),
            ],
        ),
        (
            "Memory & cache",
            [
                ("total_memory_bytes", "Device memory"),
                ("memory_bus_width_bits", "Memory bus width"),
                ("estimated_memory_bandwidth_gb_s", "Bandwidth (DDR estimate)"),
                ("l1_cache_bytes", "L1 capacity"),
                ("global_l1_cache_supported", "L1 for global memory"),
                ("local_l1_cache_supported", "L1 for local memory"),
                ("l2_cache_bytes", "L2 capacity"),
            ],
        ),
        (
            "Shared memory",
            [
                ("shared_memory_per_sm_bytes", "Per SM"),
                ("shared_memory_per_block_bytes", "Per block (default)"),
                ("shared_memory_per_block_optin_bytes", "Per block (opt-in)"),
                ("reserved_shared_memory_per_block_bytes", "Reserved / block"),
            ],
        ),
        (
            "Clocks (driver attributes)",
            [
                ("gpu_clock_khz", "GPU clock"),
                ("gpu_clock_period_ns", "GPU clock period"),
                ("memory_clock_khz", "Memory clock"),
                ("memory_clock_period_ns", "Memory clock period"),
            ],
        ),
    ]
    print(heading("NVIDIA GPU Device Report"))
    print(
        f"CUDA driver API {report['cuda_driver_api_version']}  |  "
        f"Visible: {report['visible_device_count']}  |  Selected: {len(report['devices'])}"
    )
    for device in report["devices"]:
        print()
        title = f"GPU {device['index']}: {device['name']}"
        sections = [
            (name, [(label, _format_value(key, device[key])) for key, label in fields])
            for name, fields in groups
        ]
        compute_rows = [
            (
                "CUDA FP32",
                _format_value("estimated_fp32_tflops", device["estimated_fp32_tflops"]),
            ),
            (
                "Tensor profile",
                device["tensor_throughput_profile"] or "N/A (unverified GPU model)",
            ),
            ("Tensor mode", "Dense / Sparse (2:4)"),
        ]
        for mode in _TENSOR_MODES:
            unit = "tops" if mode == "int8" else "tflops"
            values = [
                _format_value(key, device[key])
                for key in (
                    f"estimated_tensor_{mode}_dense_{unit}",
                    f"estimated_tensor_{mode}_sparse_{unit}",
                )
            ]
            compute_rows.append((f"Tensor {mode.upper()}", " / ".join(values)))
        sections.insert(1, ("Theoretical compute throughput (estimated)", compute_rows))
        label_width = max(len(label) for _, rows in sections for label, _ in rows)
        width = max(
            68,
            len(title),
            len(device["pci_bus_id"]) + 9,
            max(
                label_width + 3 + len(value)
                for _, rows in sections
                for _, value in rows
            ),
        )
        border = "+" + "-" * (width + 2) + "+"

        def row(text="", highlight=False):
            line = f"| {text:<{width}} |"
            print(heading(line) if highlight else line)

        print(border)
        row(title, highlight=True)
        row(f"PCI bus: {device['pci_bus_id']}")
        for name, rows in sections:
            print(border)
            row(name, highlight=True)
            for label, value in rows:
                row(f"{label:<{label_width}} : {value}")
        print(border)
    print("\n" + heading("Notes"))
    notes = ["GB = 10^9 bytes; GiB / MiB / KiB use powers of 1024.", *report["notes"]]
    for note in notes:
        print(
            textwrap.fill(
                note, width=88, initial_indent="  - ", subsequent_indent="    "
            )
        )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="aib check-device",
        description="Inspect NVIDIA GPU capabilities using the CUDA driver (no torch or CUDA Toolkit required).",
    )
    parser.add_argument(
        "--device",
        type=int,
        action="append",
        metavar="INDEX",
        help="CUDA device index; repeat to select several (default: all visible GPUs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON with explicit unit suffixes and null for unavailable attributes",
    )
    args = parser.parse_args(argv)
    try:
        report = collect_devices(args.device)
    except DeviceQueryError as exc:
        print(f"check-device: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
