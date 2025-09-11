import os

from ai_infra_bench import run_client

base_url = os.environ["BASE_URL"]
input_len = 1200
output_len = 800
dataset_path = os.environ["SHAREGPT_DATASET"]

client_template = """
python -m sglang.bench_serving --base-url {base_url}
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
client_cmds = [
    client_template.format(
        base_url=base_url,
        input_len=input_len,
        output_len=output_len,
        dataset_path=dataset_path,
        request_rate=rate,
        num_prompt=rate * 10,
    )
    for rate in range(2, 10, 4)
]

input_features = ["request_rate"]

metrics = [
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "output_throughput",
]

if __name__ == "__main__":
    run_client(
        client_cmds=client_cmds,
        input_features=input_features,
        metrics=metrics,
        label="exp_label",
        output_dir="output",
    )
