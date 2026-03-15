import os
from typing import List

from ai_infra_bench import client_gen

# input args
base_url = os.environ["BASE_URL"]
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

# construct client requests
client_template = """
python -m sglang.bench_serving
        --base-url {base_url}
		--backend sglang-oai
        --tokenizer Qwen/Qwen3-0.6B
        --model Qwen/Qwen3-0.6B
		--dataset-path {dataset_path}
		--dataset-name random
		--random-range-ratio 1
		--random-input-len {input_len}
		--random-output-len {output_len}
		--request-rate {request_rate}
		--max-concurrency {request_rate}
		--num-prompt {num_prompt}
"""
rate_lists: List[int] = [1, 2, 4, 8]
client_cmds: List[str] = [
    *[
        client_template.format(
            base_url=base_url,
            input_len=1200,
            output_len=800,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=min(max(rate * 10, 80), 250),  # clip to [80, 250]
        )
        for rate in rate_lists
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=800,
            output_len=1200,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=min(max(rate * 10, 80), 250),  # clip to [80, 250]
        )
        for rate in rate_lists
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=3500,
            output_len=1500,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=min(max(rate * 10, 80), 250),  # clip to [80, 250]
        )
        for rate in rate_lists
    ],
]


if __name__ == "__main__":
    client_gen(
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=output_metrics,
        server_label="qwen3_06b",
        n=3,
        only_last=True,
        output_dir="client_gen_output",
    )
