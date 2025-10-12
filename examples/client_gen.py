import os

from ai_infra_bench import client_gen

base_url = os.environ["BASE_URL"]
dataset_path = os.environ["SHAREGPT_DATASET"]

client_template = """
python -m sglang.bench_serving \
        --base-url {base_url}
		--backend sglang-oai
        --tokenizer deepseek-ai/DeepSeek-R1-0528
        --model deepseek-ai/DeepSeek-R1-0528
		--dataset-path {dataset_path}
		--dataset-name random
		--random-range-ratio 1
		--random-input-len {input_len}
		--random-output-len {output_len}
		--request-rate {request_rate}
		--max-concurrency {request_rate}
		--num-prompt {num_prompt}
"""
rate_lists = [1, 2, 4, 8, 16, 24, 32, 40]
client_cmds = [
    *[
        client_template.format(
            base_url=base_url,
            input_len=2000,
            output_len=1500,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=rate * 10,
        )
        for rate in rate_lists
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=900,
            output_len=1200,
            dataset_path=dataset_path,
            request_rate=rate,
            num_prompt=rate * 10,
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
            num_prompt=rate * 10,
        )
        for rate in rate_lists
    ],
]
print(type(client_cmds), type(client_cmds[0]))

input_features = [
    "random_input_len",
    "random_output_len",
    "request_rate",
    "max_concurrency",
]

output_metrics = [
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_itl_ms",
    "output_throughput",
    "p99_e2e_latency_ms",
]

if __name__ == "__main__":
    client_gen(
        client_cmds=client_cmds,
        input_features=input_features,
        metrics=output_metrics,
        labels="deepseek_r1",
        output_dir="ali_output",
    )
