import os
from typing import List

from ai_infra_bench.sgl import cmp_bench

# Args for server_cmds, client_cmds
input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
tp_size = 1
model_path = os.environ["QWEN332BFP8"]


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
    --kv-cache-dtype fp8_e4m3
"""

server_cmds: List[str] = [
    server_template.format(
        model_path=model_path, tp_size=tp_size, host=host, port=port
    ),
    server_template.format(model_path=model_path, tp_size=tp_size, host=host, port=port)
    + " --tool-call-parser qwen25",
]
labels = ["Qwen3-32B-FP8", "QWEN3-32B-FP8-Without-tool"]

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

client_cmds: List[str] = [
    client_template.format(
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
        request_rate=rate,
        num_prompt=rate * 10,
    )
    for rate in range(4, 12 + 1, 2)
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
    cmp_bench(
        server_cmds=server_cmds,
        client_cmds=client_cmds,
        input_features=input_features,
        metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir="tool_cmp_bench_output",
    )
