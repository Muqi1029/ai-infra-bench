import subprocess
from typing import Tuple


def get_first_gpu_info() -> Tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "N/A", "N/A"

    first_gpu = next(
        (line.strip() for line in result.stdout.splitlines() if line.strip()),
        "",
    )
    try:
        name, memory_mib = (value.strip() for value in first_gpu.rsplit(",", 1))
    except ValueError:
        return "N/A", "N/A"
    return name, f"{memory_mib} MiB"
