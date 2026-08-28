<div align="center">

![ai_infra_bench](assets/main.png)

[![LICENSE](https://img.shields.io/badge/license-Apache_2.0-orange.svg)](LICENSE)
[![PYTHON VERSION](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PYPI PROJECT](https://img.shields.io/pypi/v/ai-infra-bench?color=green)](https://pypi.org/project/ai-infra-bench/)

</div>

# AI Infra Bench

AI Infra Bench measures OpenAI-compatible inference endpoints, evaluates model
outputs, and finds the highest load that satisfies a service-level objective.
It is backend-independent and does not require a serving framework SDK.

## Install

```bash
pip install ai-infra-bench
```

Python 3.10 or newer is required.

## CLI

Send one request:

```bash
aib req --base-url http://127.0.0.1:30000 --prompt "Who are you?"
```

Run a random-token benchmark:

```bash
aib bench \
  --base-url http://127.0.0.1:30000 \
  --dataset random \
  --input-len 1024 \
  --output-len 256 \
  --random-range-ratio 0.5 \
  --num-requests 100 \
  --max-concurrency 16
```

When `--random-range-ratio` is set, each random request samples its input and
output length uniformly from the configured ratio of the target length up to
the target length. The default value of `1.0` preserves fixed-length requests.

Evaluate the text-only portion of Humanity's Last Exam directly through a
chat-completions endpoint. Local JSONL files are supported; image questions
are skipped because this adapter does not require a multimodal runtime:

```bash
aib eval-dataset \
  --evals hle \
  --dataset-path ./hle.jsonl \
  --num-shots 0 \
  --num-questions 20 \
  --base-url http://127.0.0.1:30000 \
  --model your-model
```

Other commands cover dataset evaluation, logits and hidden-state comparison,
metric plotting, and local Prometheus monitoring:

```bash
aib --help
```

Replay session-shaped JSONL payloads with one concurrency slot per session:

```bash
aib session-bench 'sessions/**/*.jsonl' \
  --max-concurrency 16 \
  --num-warmup-sessions 3
```

Requests in each file are sent in order. Different files run concurrently up
to `--max-concurrency`.

## SLO Search

SLO searches are configured in YAML. The command probes the configured range
and returns the highest `max_concurrency` or `request_rate` that satisfies every
condition:

```yaml
endpoint:
  base_url: http://127.0.0.1:30000
  api_key: EMPTY
  model: null
request:
  payload:
    messages:
      - role: user
        content: hello
benchmark:
  num_requests: 100
  warmup_requests: 5
  max_concurrency: 32
  request_rate: inf
search:
  parameter: max_concurrency
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

Run it with:

```bash
aib slo examples/slo.yaml -o slo-result.yaml
```

## Output

SLO runs write a YAML report with every probe, its metrics, and condition results.
Request benchmarks can also write machine-readable JSON or JSONL metrics for
later visualization.

## Development

```bash
python -m pip install -e .
pytest
```

Licensed under the [Apache License 2.0](LICENSE).
