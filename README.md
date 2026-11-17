<div align="center">

![ai_infra_bench](assets/main.png)

[![LICENSE](https://img.shields.io/badge/license-Apache_2.0-orange.svg)](LICENSE)
[![PYTHON VERSION](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PYPI PROJECT](https://img.shields.io/pypi/v/ai-infra-bench?color=green)](https://pypi.org/project/ai-infra-bench/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://muqi1029.github.io/ai-infra-bench/)

**[中文文档](https://muqi1029.github.io/ai-infra-bench/) · [English Docs](https://muqi1029.github.io/ai-infra-bench/en/)**

</div>

# AI Infra Bench

AI Infra Bench (`aib`) is a lightweight CLI for benchmarking OpenAI-compatible inference endpoints, evaluating correctness, and finding SLO capacity limits without serving-framework SDKs.

## Install

```bash
pip install ai-infra-bench
# Optional datasets & tokenizers:
pip install 'ai-infra-bench[data]'
# Optional safetensors weight inspection:
pip install 'ai-infra-bench[weight]'
```

Python 3.10+ is required.

## Quick Usage

### Send a Request (`aib req`)

```bash
aib req --base-url http://127.0.0.1:30000 --prompt "Who are you?"
cat payload.json | aib req --payload-path - --model your-model
cat prompt.txt | aib req --prompt - --max-tokens 128
```

### Benchmark Performance (`aib bench`)

```bash
aib bench \
  --base-url http://127.0.0.1:30000 \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 --max-concurrency 16 \
  --goodput ttft:500 tpot:50 \
  --metric-path metrics.jsonl

aib plot-metrics metrics.jsonl -o metrics.html
```

Reports TTFT, ITL, TPOT, latency percentiles, input/output/total token throughput, cache hit ratios, and Goodput.

### Session Replay & SLO Search

```bash
aib session-bench 'sessions/**/*.jsonl' --max-concurrency 16 --num-warmup-sessions 3
aib slo examples/slo.yaml -o slo-result.yaml
```

### Correctness & Diagnostics

```bash
aib eval-dataset --evals gsm8k --base-url http://127.0.0.1:30000 --model your-model --num-questions 20
aib eval-logits --base-urls http://127.0.0.1:30000 http://127.0.0.1:30001 --model your-model
aib check-device
aib check-weight --path /path/to/model
aib monitor --base-urls http://127.0.0.1:30000 --open
```

## Documentation

Read the full guides in [简体中文](https://muqi1029.github.io/ai-infra-bench/) or [English](https://muqi1029.github.io/ai-infra-bench/en/). See [website/README.md](website/README.md) to preview locally.

## Development

```bash
python -m pip install -e .
pytest
```

Licensed under the [Apache License 2.0](LICENSE).
