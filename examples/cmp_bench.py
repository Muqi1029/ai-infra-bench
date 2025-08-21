import os
from typing import List

from ai_infra_bench.sgl import cmp_bench

input_len = 1200
output_len = 800
host = "127.0.0.1"
port = "8888"
tp_size = 1


server_template = """
python -m sglang.launch_server --model-path {model_path} --tp-size {tp_size}
--host {host} --port {port}
"""

launch_sgl_server_cmds: List[str] = [
    server_template.format(
        model_path=os.environ["QWEN306B"], tp_size=tp_size, host=host, port=port
    ),
    server_template.format(
        model_path=os.environ["QWEN38B"], tp_size=tp_size, host=host, port=port
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
		--request-rate {request_rate}
		--num-prompt {num_prompt}
		--max-concurrency {request_rate}
"""
labels = ["Qwen3-0.6B-TP1", "Qwen3-8B-TP1"]

launch_sgl_client_cmds: List[str] = [
    client_template.format(
        host=host,
        port=port,
        input_len=input_len,
        output_len=output_len,
        request_rate=rate,
        num_prompt=rate * 10,
    )
    for rate in range(4, 13, 4)
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

if __name__ == "__main__":
    cmp_bench(
        server_cmds=launch_sgl_server_cmds,
        client_cmds=launch_sgl_client_cmds,
        input_features=input_features,
        metrics=metrics,
        labels=labels,
        host=host,
        port=port,
        output_dir="cmp_qwen3_0.6b_vs_8b",
    )
