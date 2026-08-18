FULL_DATA_JSON_PATH = "full_data_json"
TABLE_NAME = "table.md"
CSV_NAME = "data.csv"
WARMUP_FILE = ".warmup.json"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
GRAPH_PER_ROW = 3

# Used when command execution is skipped in CI.
demo_output = {
    "request_rate": 10.0,
    "max_concurrency": 10,
    "random_input_len": 1200,
    "random_output_len": 800,
    "p99_ttft_ms": 40.0,
    "p99_tpot_ms": 5.0,
    "p99_itl_ms": 5.0,
    "output_throughput": 100.0,
}
