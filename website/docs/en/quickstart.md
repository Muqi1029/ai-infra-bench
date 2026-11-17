# Quickstart

Requires Python 3.10+ and an OpenAI-compatible inference server (default `http://127.0.0.1:30000`).

## 1. Install

::: code-group

```bash [pip]
pip install ai-infra-bench
# Optional datasets, tokenizers & safetensors checks
pip install 'ai-infra-bench[data,weight]'
```

```bash [uv]
uv tool install ai-infra-bench
# Or install extras into the active environment
uv pip install 'ai-infra-bench[data,weight]'
```

:::

## 2. Smoke Test

Verify endpoint connectivity and authentication with a single streamed request:

```bash
aib req \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --prompt "Explain prefix caching in one sentence."
```

## 3. Benchmark & Plot

Run a concurrent random-token benchmark and export an interactive HTML report:

```bash
aib bench \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 \
  --max-concurrency 1 8 32 \
  --goodput ttft:500 tpot:50 \
  --disable-flush-cache \
  --metric-path metrics.jsonl

aib plot-metrics metrics.jsonl -o metrics.html
```

## Next Steps

- [Benchmark & SLO](/en/benchmark): random loads, prefix caching, session replay, and SLO search
- [Evaluation](/en/evaluation): dataset accuracy, Top-K logits diff, and hidden-states checks
- [Requests & Tools](/en/tools): stdin debugging, monitoring stack, GPU & weight checks
- [CLI & Metrics](/en/reference): all 14 subcommands, metric formulas, and FAQ
