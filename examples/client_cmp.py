import os

from ai_infra_bench import client_cmp
from ai_infra_bench.utils import ServerAccessInfo

# input args
input_len = 1200
output_len = 800
dataset_path = os.environ["SHAREGPT_DATASET"]
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
# don't set --base-url due to it will be contained in the server access infos
client_template = """
python -m sglang.bench_serving
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
rate_list = [1, 2, 4, 8]
client_cmds = [
    client_template.format(
        input_len=input_len,
        output_len=output_len,
        dataset_path=dataset_path,
        request_rate=rate,
        num_prompt=min(max(rate * 10, 80), 250),  # clip to [80, 250]
    )
    for rate in rate_list
]

# construct server access info
server_access_infos = [
    ServerAccessInfo(
        base_url="http://localhost:8888", api_key="JustKeepMe", label="old"
    ),
    ServerAccessInfo(
        base_url="http://localhost:8889", api_key="JustKeepMe", label="new"
    ),
]


if __name__ == "__main__":
    client_cmp(
        server_access_infos=server_access_infos,
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=output_metrics,
        n=3,
        only_last=True,
        output_dir="version_cmp_bench",
    )
