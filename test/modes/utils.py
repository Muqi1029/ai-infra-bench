import os

from ai_infra_bench.utils import CSV_NAME, FULL_DATA_JSON_PATH, TABLE_NAME, WARMUP_FILE

input_features = [
    "random_input_len",
    "random_output_len",
    "request_rate",
    "max_concurrency",
]

output_metrics = [
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "output_throughput",
    "p99_e2e_latency_ms",
]


def check_output_content(output_dir, expected_files=None):
    expected_files = expected_files or [
        FULL_DATA_JSON_PATH,
        TABLE_NAME,
        CSV_NAME,
        WARMUP_FILE,
    ]
    for f in expected_files:
        assert os.path.exists(os.path.join(output_dir, f)), f"Missing {f}"
