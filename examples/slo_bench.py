import sys

sys.path.append("..")
import os
from typing import List, Tuple

from ai_infra_bench.sgl import slo_bench

input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
tp_size = 2


server_template = """
python -m sglang.launch_server --model-path {model_path} --tp-size {tp_size} 
--host {host} --port {port} --kv-cache-dtype fp8_e4m3
"""

launch_sgl_server_cmds: List[str] = [
    server_template.format(
        model_path=os.environ["QWEN330BA3BFP8"], tp_size=tp_size, host=host, port=port
    ),
]

client_template = """
python -m sglang.bench_serving --host {host} --port {port}
		--backend sglang-oai
		--dataset-path /root/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
		--dataset-name random
		--random-range-ratio 1
		--random-input-len {input_len}
		--random-output-len {output_len}
"""
labels = ["QWEN330BA3BFP8-TP2"]

launch_sgl_client_cmds: List[str] = [
    client_template.format(
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
    )  # cannot set request_rate
]
request_rates: List[Tuple[int, int]] = [
    (10, 80),
]

input_features = [
    "request_rate",
]
metrics = [
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "output_throughput",
]  # used to plot, make table


def check_slo(item):
    return (
        item["p99_ttft_ms"] < 3000
        and item["p99_tpot_ms"] < 100
        and item["p99_itl_ms"] < 100
    )


if __name__ == "__main__":
    slo_bench(
        launch_sgl_server_cmds,
        launch_sgl_client_cmds,
        request_rates=request_rates,
        input_features=input_features,
        metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir=f"slo_output_{labels[0]}",
        check_slo=check_slo,
    )
