
<div align="center">

![ai_infra_bench](assets/main.png)
  [![LICENSE](https://img.shields.io/badge/license-Apache_2.0-orange.svg)](https://github.com/Muqi1029/ai_infra_bench/blob/main/LICENSE) [![PYTHON VERSION](https://img.shields.io/badge/python-3.10+-blue)](https://img.shields.io/badge/python-3.10+-blue) [![PYPI PROJECT](https://img.shields.io/pypi/v/ai-infra-bench?color=green)]( https://pypi.org/project/ai-infra-bench/)
</div>

# Motivation

As large language model (LLMs) grow more capable, industries are eager to deploy them locally to ensure data security, reduce costs, and integrate them into daily workflows -- such as building agents that can perform comprehensive analyses to improve efficiency.

Thus, the question of how to deploy LLMs efficiently has become a critical challenge for many companies. Here, we focus solely on the benchmarking perspective -- while workload-specific optimization is a core topic, we leave it aside for simplicity as it is beyond the scope of this project. Practitioners in AI infrastructure often need to benchmark models or deployment startegies to determine whether they meet the service-level objectives (SLOs) of a particular business.

Benchmarking is often done by repeatedly running benchmark scripts (e.g., SGLang's `bench_serving`) and manually copying the resulting metrics into tables or plots for comparison. This process is tedious, error-prone, and highly repetitive. An alternative is to write custom shell scripts, but they are difficult to reuse and lack the flexibility and expressive power of modern programming language like Python.

This project aims to free AI infrastructure engineers from repetitive benchmarking tasks. Simply define your server and client launch scripts in a single python file, and the tool will automatically generate clear Markdown tables and elegant HTML plots. The entire process is fully automated -- you can focus on other work while the script run!

# Overview

There are totally 3 modes when benchmarking from the viewpoint of ai infra workers

1. **General**: Evaluate the performance of a single deployment across various workloads.
2. **Cmp**: Compare the performance of multiple deployment options in the same workload.
3. **SLO**: Identify the most demanding workload that still meets the required service-level objectives given a deployment option.

This project automatically generates benchmarking results as clean Markdown tables and interactive HTML graphs for clear visualization. Below are some example outputs.

## General Bench

### Table

Title: **Qwen3-8B-TP1**
| request_rate |     | p99_ttft_ms | p99_tpot_ms | p99_itl_ms | output_throughput |
| ------------ | --- | ----------- | ----------- | ---------- | ----------------- |
| 4.00         |     | 77.49       | 24.41       | 24.86      | 163.37            |
| 8.00         |     | 1145.30     | 27.41       | 26.60      | 298.41            |
| 12.00        |     | 2290.93     | 29.66       | 28.42      | 399.15            |
| 16.00        |     | 3100.51     | 32.12       | 30.06      | 492.53            |

### Plot

![General Bench](./assets/gen_bench.png)

## Cmp Bench

### Table
Metric: **p99_ttft_ms**
| request_rate |     | Qwen3-32B-FP8-With-CUDAGRAPH | QWEN3-32B-FP8-Without-CUDAGRAPH |
| ------------ | --- | ---------------------------- | ------------------------------- |
| 12.00        |     | 54963.08                     | 58177.51                        |
| 16.00        |     | 55300.19                     | 58545.89                        |
| 20.00        |     | 105988.45                    | 112344.49                       |
| 24.00        |     | 106190.07                    | 112756.17                       |
| 28.00        |     | 157047.98                    | 166668.61                       |
| 32.00        |     | 157452.81                    | 167846.66                       |
| 36.00        |     | 208123.38                    | 221496.96                       |





Metric: **p99_tpot_ms**
| request_rate |     | Qwen3-32B-FP8-With-CUDAGRAPH | QWEN3-32B-FP8-Without-CUDAGRAPH |
| ------------ | --- | ---------------------------- | ------------------------------- |
| 12.00        |     | 60.17                        | 65.90                           |
| 16.00        |     | 59.42                        | 63.78                           |
| 20.00        |     | 59.42                        | 63.63                           |
| 24.00        |     | 59.42                        | 63.73                           |
| 28.00        |     | 59.42                        | 63.93                           |
| 32.00        |     | 59.42                        | 64.35                           |
| 36.00        |     | 59.42                        | 64.57                           |





Metric: **p99_itl_ms**
| request_rate |     | Qwen3-32B-FP8-With-CUDAGRAPH | QWEN3-32B-FP8-Without-CUDAGRAPH |
| ------------ | --- | ---------------------------- | ------------------------------- |
| 12.00        |     | 59.39                        | 64.51                           |
| 16.00        |     | 59.33                        | 64.13                           |
| 20.00        |     | 59.33                        | 64.17                           |
| 24.00        |     | 59.33                        | 64.41                           |
| 28.00        |     | 59.34                        | 64.38                           |
| 32.00        |     | 59.33                        | 64.59                           |
| 36.00        |     | 59.34                        | 64.47                           |





Metric: **output_throughput**
| request_rate |     | Qwen3-32B-FP8-With-CUDAGRAPH | QWEN3-32B-FP8-Without-CUDAGRAPH |
| ------------ | --- | ---------------------------- | ------------------------------- |
| 12.00        |     | 125.32                       | 117.98                          |
| 16.00        |     | 125.33                       | 118.70                          |
| 20.00        |     | 125.35                       | 118.77                          |
| 24.00        |     | 125.50                       | 118.69                          |
| 28.00        |     | 125.35                       | 118.36                          |
| 32.00        |     | 125.35                       | 118.45                          |
| 36.00        |     | 125.34                       | 118.31                          |


### Plot
![Cmp Bench](assets/cmp_bench.png)

## Slo Bench

### Table
Title: **QWEN3-30B-A3B-FP8-TP1**
| request_rate |     | p99_ttft_ms | p99_tpot_ms | p99_itl_ms | output_throughput |
| ------------ | --- | ----------- | ----------- | ---------- | ----------------- |
| 27.00        |     | 1813.80     | 41.61       | 42.12      | 652.56            |
| 36.00        |     | 2482.11     | 46.24       | 46.36      | 789.62            |
| 40.00        |     | 2958.86     | 46.18       | 46.33      | 874.70            |
| 42.00        |     | 2876.21     | 48.62       | 48.16      | 871.91            |
| 43.00        |     | 2889.78     | 48.59       | 48.16      | 895.68            |
| 44.00        |     | 3080.99     | 48.36       | 48.07      | 912.25            |
| 45.00        |     | 3146.45     | 48.54       | 47.99      | 935.28            |

### Plot
![slo bench](assets/slo_bench.png)

# Install

```py
pip install ai-infra-bench
```

# How to use

Concrete usage examples and argument configurations can be found in the [examples subdirectory](./examples)

# Limitation

The following limitations also represent the project's TODO items for improving usability:

1. Coverage

   - Currently only supports SGLang.

   - Server and client launch scripts must strictly start with `python -m sglang.launch_server` or `python -m sglang.bench_serving` for security reasons. This restriction prevents accidentally executing unsafe scripts.

2. Output content hardcoding and inflexibility

   - JSON metrics generated by `bench_serving` are hardcoded. Users cannot customize file names, table titles, or graph labels, which may cause confusion.
   - The output directory must not exist before running the benchmark to ensure a clean workspace.
   - The contents of tables and plots are fixed for the three benchmarking modes and cannot yet be customized.
