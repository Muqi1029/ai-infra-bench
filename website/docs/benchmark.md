# 压测与 SLO

`aib bench`、`aib session-bench` 与 `aib slo` 用于测量真实推理吞吐、延迟分布与 SLO 容量上限。

## 性能压测 (`aib bench`)

选择随机长度、内置数据集或自定义请求体发起并发压测：

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
# 匹配多个 JSON / JSONL 文件
aib bench --payload-path 'requests/**/*.jsonl' --max-concurrency 16 --disable-flush-cache

# 从 stdin 读取单个请求模板回放
cat request.json | aib bench --payload-path - --num-requests 50 --disable-flush-cache
```

:::

| 关键参数 | 作用 |
| --- | --- |
| `--max-concurrency 1 8 32` | 传入多组并发依次压测并合并汇总 |
| `--goodput ttft:500 tpot:50` | 统计满足全部毫秒阈值的有效吞吐（支持 `ttft`、`tpot`、`itl`、`e2el`） |
| `--cache-ratio 0.5` | 所有请求共享 50% 前缀 token 并在正式压测前自动预热 |
| `--random-range-ratio 0.5` | 请求长度在 `[0.5 * target, target]` 区间均匀采样（默认 `1.0`） |
| `--num-warmup-requests 0` | 关闭预热请求（默认预热 `10` 条） |
| `--disable-flush-cache` | 通用非 SGLang 服务跳过 `/flush_cache` 调用 |

## 多轮会话回放 (`aib session-bench`)

每个 JSONL 文件为一个独立会话：会话内按行串行推进，会话间按 `--max-concurrency` 并发。

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

## SLO 二分容量搜索 {#slo}

在整数区间内二分搜索满足全部 SLO 条件的最大 `max_concurrency` 或 `request_rate`。

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
  parameter: max_concurrency   # 或 request_rate
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

```bash [运行探测]
aib slo slo.yaml -o slo-result.yaml
```

:::

- 也可将 `request` 替换为 `payload: {messages: [...]}` 使用 chat 接口；可用 `metric` 列表见[命令与指标参考](/reference#metrics)。

## 可视化对比报告 (`aib plot-metrics`)

多次运行指定 `--label` 追加到同一 `.jsonl` 文件，一键生成交互式 Plotly 延迟/吞吐曲线：

```bash
aib bench --dataset random --num-requests 100 --metric-path metrics.jsonl --label baseline
aib bench --dataset random --num-requests 100 --metric-path metrics.jsonl --label fp8
aib plot-metrics metrics.jsonl -o metrics.html
```
