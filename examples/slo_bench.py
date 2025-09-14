import os
from typing import Dict, List, Tuple

from ai_infra_bench.sgl import slo_bench

input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
tp_size = 1
qwen3_30b_a3b_fp8_model_path = os.environ["QWEN3_30B_A3B_FP8"]
dataset_path = os.environ["SHAREGPT_DATASET"]


####################################
# Constructing server_cmds & labels
####################################
server_template = """
python -m sglang.launch_server --model-path {model_path} --tp-size {tp_size}
--host {host} --port {port} --kv-cache-dtype fp8_e4m3
"""

server_cmds: List[str] = [
    server_template.format(
        model_path=qwen3_30b_a3b_fp8_model_path, tp_size=tp_size, host=host, port=port
    ),
]
labels = ["QWEN3-30B-A3B-FP8-TP1"]

##########################
# Constructing client_cmds
##########################
client_template = """
python -m sglang.bench_serving --host {host} --port {port}
		--backend sglang-oai
		--dataset-path {dataset_path}
		--dataset-name random
		--random-range-ratio 1
		--random-input-len {input_len}
		--random-output-len {output_len}
"""

client_cmds: List[str] = [
    client_template.format(
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
        dataset_path=dataset_path,
    )  # NOTE: cannot set request_rate and max-concurrency
]

#####################
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


def check_slo(item: Dict) -> bool:
    return (
        item["p99_ttft_ms"] < 3000
        and item["p99_tpot_ms"] < 100
        and item["p99_itl_ms"] < 100
    )


if __name__ == "__main__":
    slo_bench(
        server_cmds=server_cmds,
        client_cmds=client_cmds,
        request_rates=request_rates,
        input_features=input_features,
        metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir="slo_bench_output",
        check_slo=check_slo,
    )
