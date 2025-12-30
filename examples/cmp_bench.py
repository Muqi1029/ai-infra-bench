import os
from typing import List

from ai_infra_bench.sgl import cmp_bench

# Args for server_cmds, client_cmds
input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
tp_size = 1
model_path = "Qwen/Qwen3-0.6B"
rate_list = [8]
dataset_path = os.environ["SHAREGPT_DATAPATH"]
input_features = [
    "random_input_len",
    "random_output_len",
    "request_rate",
    "max_concurrency",
]

output_metrics = [
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "output_throughput",
]


####################################
# Constructing server_cmds & labels
####################################
server_template = """
python -m sglang.launch_server
    --model-path {model_path}
    --tp-size {tp_size}
    --host {host}
    --port {port}
    --disable-radix-cache
"""

server_cmds: List[str] = [
    server_template.format(
        model_path=model_path, tp_size=tp_size, host=host, port=port
    ),
    server_template.format(model_path=model_path, tp_size=tp_size, host=host, port=port)
    + " --tool-call-parser qwen",
]
server_labels = ["Qwen3-06B", "QWEN3-06B-With-Tool-Call-Parser"]

##########################
# Constructing client_cmds
##########################
client_template = """
python -m sglang.bench_serving
        --host {host}
        --port {port}
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

client_cmds: List[str] = [
    client_template.format(
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
        request_rate=rate,
        dataset_path=dataset_path,
        num_prompt=min(max(rate * 10, 80), 250),
    )
    for rate in rate_list
]

#####################

if __name__ == "__main__":
    cmp_bench(
        server_cmds=server_cmds,
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=output_metrics,
        server_labels=server_labels,
        host=host,
        port=port,
        n=1,
        only_last=True,
        output_dir="tool_cmp_bench_output",
        disable_warmup=True,
    )
