import os
from typing import List

from ai_infra_bench.sgl import gen_bench

# Args for server_cmds, client_cmds
input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
qwen3_0_6b_model_path = os.environ["QWEN306B"]
qwen3_8b_model_path = os.environ["QWEN38B"]
dataset_path = os.environ["S_GPT_DATASET"]


####################################
# Constructing server_cmds & labels
####################################
server_template = """
python -m sglang.launch_server --model-path {model_path} --tp-size {tp_size}
--host {host} --port {port}
"""

server_cmds: List[str] = [
    server_template.format(
        model_path=qwen3_0_6b_model_path, tp_size=1, host=host, port=port
    ),
    server_template.format(
        model_path=qwen3_8b_model_path, tp_size=1, host=host, port=port
    ),
]
labels = ["Qwen3-0.6B-TP1", "Qwen3-8B-TP1"]

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
		--request-rate {request_rate}
		--num-prompt {num_prompt}
		--max-concurrency {request_rate}
"""

client_cmds: List[List[str]] = [
    [
        client_template.format(
            host=host,
            port=port,
            input_len=input_len,
            output_len=output_len,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=rate * 10,
        )
        for rate in range(10, 32, 4)
    ],
    [
        client_template.format(
            host=host,
            port=port,
            input_len=input_len,
            output_len=output_len,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=rate * 10,
        )
        for rate in range(4, 17, 4)
    ],
]

#####################

input_features = [
    "request_rate",
]
metrics = [
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "output_throughput",
]

if __name__ == "__main__":
    gen_bench(
        server_cmds=server_cmds,
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir="general_bench_output",
    )
