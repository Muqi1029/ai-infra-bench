client_cmd_str = """
python -m sglang.bench_serving \
        --base-url http://localhost:8888
		--backend sglang-oai
        --tokenizer Qwen/Qwen3-0.6B
        --tokenizer Qwen/Qwen3-0.6B
		--dataset-name random
		--random-range-ratio 1
		--random-input-len 1200
		--random-output-len 800
		--request-rate 10
		--max-concurrency 10
		--num-prompt 40
"""

server_cmd_str = f"""
python -m sglang.launch_server
    --model-path Qwen/Qwen3-0.6B
    --port 8888
"""

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
