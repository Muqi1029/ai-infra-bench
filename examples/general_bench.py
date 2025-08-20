import sys

sys.path.append("..")
import os
from typing import List

from ai_infra_bench.sgl import general_bench

# Args for server_cmds, client_cmds
input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"


####################################
# Constructing server_cmds & labels
####################################
server_template = """
python -m sglang.launch_server --model-path {model_path} --tp-size {tp_size}
--host {host} --port {port}
"""

server_cmds: List[str] = [
    server_template.format(
        model_path=os.environ["QWEN306B"], tp_size=1, host=host, port=port
    ),
    server_template.format(
        model_path=os.environ["QWEN38B"], tp_size=1, host=host, port=port
    ),
]
labels = ["Qwen3-0.6B-TP1", "Qwen3-8B-TP1"]

##########################
# Constructing client_cmds
##########################
client_template = """
python -m sglang.bench_serving --host {host} --port {port}
		--backend sglang-oai
		--dataset-path /root/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
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
            request_rate=rate,
            num_prompt=rate * 10,
        )
        for rate in range(4, 12, 2)
    ],
    [
        client_template.format(
            host=host,
            port=port,
            input_len=input_len,
            output_len=output_len,
            request_rate=rate,
            num_prompt=rate * 10,
        )
        for rate in range(10, 13, 2)
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
    general_bench(
        server_cmds=server_cmds,
        client_cmds=client_cmds,
        input_features=input_features,
        metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir="general_output",
    )
