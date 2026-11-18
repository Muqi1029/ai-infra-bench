# 命令与指标参考

使用 `aib <subcommand> --help` 查看完整实时参数列表；全部 `--help` 均采用零重依赖懒加载。

## 全部子命令

| 命令 | 用途 |
| --- | --- |
| `req` | 发送单条请求（支持 `-` 标准输入与 SSE 调试） |
| `bench` | 闭环并发 / 开环 `--qps` 性能压测与 Goodput 统计 |
| `session-bench` | 会话级并发多轮回放（别名 `session-reply-bench`） |
| `slo` | YAML 驱动的 SLO 二分容量搜索（`max_concurrency` / `qps`） |
| `plot-metrics` | 从 YAML/JSONL 导出含 Pareto 前沿与标签筛选的交互式 HTML 报告 |
| `eval-dataset` | GSM8K / GPQA / AIME25 / HLE / DeepSWE 评测 |
| `eval-logits` | 多端点首 token Top-K logprobs 对齐校验 |
| `eval-hidden-states` | 多端点隐藏状态对齐校验 |
| `monitor` | 跨集群 Prometheus + Grafana 监控中心与 `--server ... -- <cmd>` 零侵入自动注册 |
| `check-device` | 零第三方依赖检测 GPU 型号、显存与驱动 |
| `check-weight` | 统计 safetensors / `.bin` 权重参数量 |
| `token` | Hugging Face tokenizer 编码/解码（别名 `tokenize`） |
| `profile-sgl` | 触发或停止 SGLang profiling |
| `reply` | 回放记录并导出 input / output token IDs |

## 通用与压测常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:30000` | 服务根地址（`eval-logits` 等多端点用 `--base-urls`） |
| `--api-key` | `OPENAI_API_KEY` / `JustKeepMe` | Bearer Token 认证密钥 |
| `--model` | 未设置 | 请求模型标识 |
| `--override-payload` | 未设置 | 覆盖顶层请求字段的 JSON 字符串 |
| `--dataset` | `random` | `random` / `gsm8k` / `gpqa` / `sharegpt` |
| `--payload-path` | 未设置 | JSON / JSONL / Glob 或 `-`（stdin） |
| `--input-len` / `--output-len` | `8192` / `1024` | 目标输入/输出 token 长度 |
| `--num-requests` | `100` | 正式压测总请求数 |
| `--max-concurrency` | `32`（`--qps` 下无限） | 并发上限（支持传入多个值顺序扫测） |
| `--qps` | `inf` | 开环每秒请求到达速率（默认无并发上限，支持传入多个值顺序扫测） |
| `--burstiness` | `1.0` | 开环到达突发系数（`1.0` 泊松分布，`0` 匀速固定间隔，`>0` Gamma 分布） |
| `--num-warmup-requests` | `10` | 预热请求数（设为 `0` 关闭） |
| `--goodput` | 未设置 | Goodput SLO 阈值，如 `ttft:500 tpot:50` |
| `--disable-flush-cache` | 关闭 | 跳过 `/flush_cache` 调用 |
| `--metric-path` | `aib-metric.yaml` | 聚合指标输出路径（`.yaml` 多文档追加，兼容 `.json` 覆盖 / `.jsonl` 追加） |

## 压测核心指标 {#metrics}

| 指标 | 定义 |
| --- | --- |
| `TTFT` | 请求发出至收到首个非空内容 / 推理 / 工具调用增量的耗时（ms） |
| `ITL` | 首个增量之后相邻流式 chunk 之间的间隔（ms） |
| `TPOT` | `(latency_ms - ttft_ms) / (completion_tokens - 1)`（ms/token） |
| `Latency` | 单次 HTTP 请求网络耗时，不含客户端排队等待（ms） |
| `Input / Output / Total throughput` | 成功请求的输入 / 输出 / 总 token 数除以压测总时长（tokens/s） |
| `Request goodput` | 同时满足 `--goodput` 全部毫秒阈值的有效请求速率与占比 |
| `Cached tokens` | 服务端返回的前缀缓存命中 token 数 |

**SLO `conditions[].metric` 可选键**：
- 计数与吞吐：`total_requests`、`successful_requests`、`failed_requests`、`success_rate`、`duration_s`、`max_concurrency`、`request_rate`、`request_throughput`、`input_throughput`、`output_throughput`、`total_token_throughput`
- 百分位分布：`mean_` / `p50_` / `p95_` / `p99_` 拼接 `ttft_ms`、`tpot_ms`、`itl_ms`、`latency_ms`、`e2e_latency_ms`、`prompt_tokens`、`completion_tokens`、`reasoning_tokens`、`cached_tokens`

## 常见问题

| 现象 | 排查方式 |
| --- | --- |
| 连接拒绝或 404 | 检查 `--base-url` 是否为根地址或 `/v1`；随机 token 使用 `/v1/completions` |
| 401 / 403 或 Model not found | 显式传入 `--api-key` 与服务端注册的 `--model` |
| 通用服务压测缓慢或报错 `/flush_cache` | 加上 `--disable-flush-cache`（遇到 HTTP 404/405 也会自动快速跳过） |
| `eval-dataset` 或 ShareGPT 提示缺包 | 运行 `pip install 'ai-infra-bench[data]'`；ShareGPT 指定长度需传 `--tokenizer` 或 `--model` |
| `session-bench` 无正式样本结果 | 默认跳过前 3 个会话预热，样本较少时添加 `--num-warmup-sessions 0` |
