import os
from typing import Dict

from ai_infra_bench import client_slo

# input args
input_len = 1200
output_len = 800
base_url = os.environ["BASE_URL"]
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
# don't set --request-rate because it is a variable in slo situation
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
"""
client_cmds = client_template.format(
    base_url=base_url,
    dataset_path=dataset_path,
    input_len=input_len,
    output_len=output_len,
)


def check_slo(item: Dict) -> bool:
    return (
        item["p99_ttft_ms"] < 3000
        and item["p99_tpot_ms"] < 100
        and item["p99_itl_ms"] < 100
    )


request_rates = [(20, 70)]


if __name__ == "__main__":
    client_slo(
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=output_metrics,
        check_slo=check_slo,
        request_rates=request_rates,
        n=3,
        only_last=True,
        output_dir="client_slo_output",
    )
