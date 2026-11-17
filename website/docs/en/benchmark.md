# Benchmark & SLO

`aib bench`, `aib session-bench`, and `aib slo` measure serving throughput, latency percentiles, Goodput, and SLO capacity bounds.

## Throughput Benchmark (`aib bench`)

Pick synthetic tokens, built-in datasets, or custom JSON/JSONL replays:

::: code-group

```bash [Random Tokens]
aib bench \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 \
  --max-concurrency 1 8 32 \
  --repeat 2 \
  --goodput ttft:500 tpot:50 \
  --metric-path metrics.jsonl
```

```bash [ShareGPT / GPQA]
aib bench \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --dataset sharegpt \
  --input-len 1024 --output-len 256 \
  --num-requests 100 \
  --disable-flush-cache
```

```bash [JSONL / Stdin]
# Replay multiple JSON / JSONL files via glob
aib bench --payload-path 'requests/**/*.jsonl' --max-concurrency 16 --disable-flush-cache

# Replay a single payload template piped from stdin
cat request.json | aib bench --payload-path - --num-requests 50 --disable-flush-cache
```

:::

| Key Flag | Purpose |
| --- | --- |
| `--max-concurrency 1 8 32` | Sweep multiple concurrency levels sequentially in one run |
| `--goodput ttft:500 tpot:50` | Report request rate & percentage meeting all millisecond thresholds (`ttft`, `tpot`, `itl`, `e2el`) |
| `--cache-ratio 0.5` | Share 50% prefix tokens across requests and warm up the prefix automatically |
| `--random-range-ratio 0.5` | Sample lengths uniformly within `[0.5 * target, target]` (default `1.0`) |
| `--num-warmup-requests 0` | Disable warmup requests (defaults to `10`) |
| `--disable-flush-cache` | Skip `/flush_cache` calls on non-SGLang endpoints |

## Multi-Turn Session Replay (`aib session-bench`)

Each JSONL file defines one conversation; turns execute sequentially within a file while sessions run concurrently up to `--max-concurrency`.

```jsonl
{"messages":[{"role":"user","content":"My name is Alex."}],"max_tokens":64}
{"messages":[{"role":"user","content":"My name is Alex."},{"role":"assistant","content":"Hello, Alex!"},{"role":"user","content":"What is my name?"}],"max_tokens":64}
```

```bash
aib session-bench 'sessions/**/*.jsonl' \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --max-concurrency 16 \
  --num-warmup-sessions 0 \
  --disable-flush-cache \
  --metric-path session-metrics.jsonl
```

## SLO Binary Search (`aib slo`) {#slo}

Binary-search the highest `max_concurrency` or `request_rate` satisfying all declarative constraints:

::: code-group

```yaml [slo.yaml]
endpoint:
  base_url: http://127.0.0.1:30000
  api_key: EMPTY
  model: your-model
request:
  input_len: 1024
  output_len: 256
benchmark:
  num_requests: 100
  warmup_requests: 5
  max_concurrency: 32
  request_rate: inf
search:
  parameter: max_concurrency   # or request_rate
  min: 1
  max: 64
conditions:
  - metric: success_rate
    operator: ">="
    value: 0.99
  - metric: p99_latency_ms
    operator: "<"
    value: 3000
```

```bash [Run Search]
aib slo slo.yaml -o slo-result.yaml
```

:::

- Replace `request` with `payload: {messages: [...]}` for chat payloads; see [CLI & Metrics](/en/reference#metrics) for supported metric keys.

## Interactive Plots (`aib plot-metrics`)

Append labeled runs into one JSONL file to compare latency and throughput curves:

```bash
aib bench --dataset random --num-requests 100 --metric-path metrics.jsonl --label baseline
aib bench --dataset random --num-requests 100 --metric-path metrics.jsonl --label fp8
aib plot-metrics metrics.jsonl -o metrics.html
```
