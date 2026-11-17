---
layout: home

hero:
  name: AI Infra Bench
  text: Benchmark, Evaluate & Size LLM Endpoints
  tagline: Zero serving-SDK lock-in. One aib CLI for throughput, Goodput, SLO binary search, accuracy alignment, and hardware checks.
  image:
    src: /logo.svg
    alt: AI Infra Bench
  actions:
    - theme: brand
      text: Get Started →
      link: /en/quickstart
    - theme: alt
      text: CLI Reference
      link: /en/reference
    - theme: alt
      text: GitHub
      link: https://github.com/Muqi1029/ai-infra-bench

features:
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>'
    title: Throughput & Goodput
    details: Measure TTFT, ITL, TPOT, token throughput, and Goodput across random tokens, datasets, or multi-turn sessions.
    link: /en/benchmark
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m16 12-4-4-4 4"/><path d="M12 16V8"/></svg>'
    title: SLO Capacity Search
    details: Binary-search the maximum max_concurrency or request_rate that satisfies declarative YAML latency targets.
    link: /en/benchmark#slo
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>'
    title: Accuracy & Logits
    details: Run GSM8K, GPQA, AIME25, and HLE benchmarks, or diff Top-K logprobs and hidden states across endpoints.
    link: /en/evaluation
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>'
    title: Requests, Monitor & GPU
    details: Debug single stdin payloads, launch local Prometheus + Grafana dashboards, and inspect GPUs or safetensors.
    link: /en/tools
---

<div class="home-preview">
  <h2 class="home-preview-title">30-Second Quickstart</h2>

::: code-group

```bash [Install]
pip install ai-infra-bench
# Optional datasets, tokenizers & weight checks
pip install 'ai-infra-bench[data,weight]'
```

```bash [Benchmark]
aib bench \
  --base-url http://127.0.0.1:30000 \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 --max-concurrency 1 8 32 \
  --goodput ttft:500 tpot:50 \
  --metric-path metrics.jsonl

aib plot-metrics metrics.jsonl -o metrics.html
```

```bash [Evaluate]
aib eval-dataset \
  --evals gsm8k \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --num-questions 50
```

:::

</div>
