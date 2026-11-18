---
layout: home

hero:
  name: AI Infra Bench
  text: 推理服务压测、评测与容量定界
  tagline: 零推理框架依赖，一条 aib 命令完成性能压测、SLO 搜索、正确性对齐与硬件诊断。
  image:
    src: /logo.svg
    alt: AI Infra Bench
  actions:
    - theme: brand
      text: 快速开始 →
      link: /quickstart
    - theme: alt
      text: 命令速查
      link: /reference
    - theme: alt
      text: GitHub
      link: https://github.com/Muqi1029/ai-infra-bench

features:
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>'
    title: 压测与 Goodput
    details: 精确测量 TTFT、ITL、TPOT、吞吐量与 Goodput 达标率，支持随机 token、数据集与多轮会话回放。
    link: /benchmark
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m16 12-4-4-4 4"/><path d="M12 16V8"/></svg>'
    title: SLO 容量搜索
    details: 通过 YAML 声明延迟与成功率约束，二分探测单实例最优 max_concurrency 或 request_rate。
    link: /benchmark#slo
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>'
    title: 正确性与精度对齐
    details: 内置 GSM8K、GPQA、AIME25、HLE 评测及多端点 Top-K logprobs / Hidden States 对齐。
    link: /evaluation
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>'
    title: 请求、监控与诊断
    details: 单请求管道调试、Prometheus + Grafana 本地监控、零依赖 GPU 检测与 safetensors 参数校验。
    link: /tools
---

<div class="home-preview">
  <h2 class="home-preview-title">30 秒上手</h2>

::: code-group

```bash [安装]
pip install ai-infra-bench
# 可选：数据集、分词器与权重检查
pip install 'ai-infra-bench[data,weight]'
```

```bash [性能压测]
aib bench \
  --base-url http://127.0.0.1:30000 \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 --max-concurrency 1 8 32 \
  --goodput ttft:500 tpot:50 \
  --metric-path metrics.yaml

aib plot-metrics metrics.yaml -o metrics.html
```

```bash [正确性评测]
aib eval-dataset \
  --evals gsm8k \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --num-questions 50
```

:::

</div>
