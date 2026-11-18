# CLI & Metrics

Run `aib <subcommand> --help` for full live flags; all `--help` handlers load with stdlib-only imports.

## Subcommands

| Command | Purpose |
| --- | --- |
| `req` | Send a single request (supports `-` stdin and raw SSE output) |
| `bench` | Concurrent performance benchmark and Goodput evaluation |
| `session-bench` | Multi-turn session replay (alias `session-reply-bench`) |
| `slo` | YAML-driven SLO binary capacity search |
| `plot-metrics` | Export interactive Plotly comparison report |
| `eval-dataset` | Accuracy evaluation (`gsm8k`, `gpqa`, `aime25`, `hle`, `constrained_decoding`, `deepswe`) |
| `eval-logits` | Multi-endpoint Top-K logprobs numerical diff |
| `eval-hidden-states` | Multi-endpoint hidden states comparison |
| `monitor` | Cross-cluster Prometheus + Grafana server with auto-registration & NAT push |
| `check-device` | Stdlib-only NVIDIA GPU & VRAM inspection |
| `check-weight` | Lightweight safetensors / `.bin` parameter audit |
| `token` | Hugging Face tokenizer encode/decode (alias `tokenize`) |
| `profile-sgl` | Start or stop SGLang profiling |
| `reply` | Replay records and export input/output token IDs |

## Common & Benchmark Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:30000` | Endpoint URL (`--base-urls` for diff commands) |
| `--api-key` | `OPENAI_API_KEY` / `JustKeepMe` | Bearer token |
| `--model` | Unset | Model identifier sent in request payload |
| `--override-payload` | Unset | JSON object overriding top-level request fields |
| `--dataset` | `random` | `random` / `gsm8k` / `gpqa` / `sharegpt` |
| `--payload-path` | Unset | JSON / JSONL / glob or `-` (stdin) |
| `--input-len` / `--output-len` | `8192` / `1024` | Target prompt and generation token counts |
| `--num-requests` | `100` | Measured request count |
| `--max-concurrency` | `32` | One or more concurrency levels to sweep |
| `--num-warmup-requests` | `10` | Warmup request count (`0` to disable) |
| `--goodput` | Unset | SLO thresholds in ms, e.g. `ttft:500 tpot:50` |
| `--disable-flush-cache` | Off | Skip `/flush_cache` calls |
| `--metric-path` | `aib-metric.jsonl` | Summary output (`.json` overwrites, `.jsonl` appends) |

## Benchmark Metrics {#metrics}

| Metric | Definition |
| --- | --- |
| `TTFT` | Elapsed ms from request dispatch to first non-empty content, reasoning, or tool chunk |
| `ITL` | Inter-chunk latency (ms) between consecutive deltas after TTFT |
| `TPOT` | `(latency_ms - ttft_ms) / (completion_tokens - 1)` (ms/token) |
| `Latency` | Per-request HTTP duration excluding client queueing (ms) |
| `Input / Output / Total throughput` | Sum of prompt / completion / total tokens over total run duration (tokens/s) |
| `Request goodput` | Throughput and ratio of requests meeting every `--goodput` threshold |
| `Cached tokens` | Prefix-cache hit token count returned by the server |

**SLO `conditions[].metric` keys**:
- Counts & throughput: `total_requests`, `successful_requests`, `failed_requests`, `success_rate`, `duration_s`, `max_concurrency`, `request_rate`, `request_throughput`, `input_throughput`, `output_throughput`, `total_token_throughput`
- Percentile stats: `mean_` / `p50_` / `p95_` / `p99_` + `ttft_ms`, `tpot_ms`, `itl_ms`, `latency_ms`, `e2e_latency_ms`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_tokens`

## FAQ

| Issue | Resolution |
| --- | --- |
| Connection refused / 404 | Check `--base-url` root vs `/v1`; random-token loads post to `/v1/completions` |
| 401 / 403 or Model not found | Pass explicit `--api-key` and server-registered `--model` |
| Non-SGLang `/flush_cache` warnings | Add `--disable-flush-cache` (HTTP 404/405 is also skipped automatically) |
| Missing dataset or tokenizer package | Install extras with `pip install 'ai-infra-bench[data]'` |
| `session-bench` skips all files | By default the first 3 sessions warm up the server; pass `--num-warmup-sessions 0` |
