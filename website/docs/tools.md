# 请求与诊断工具

覆盖单请求管道调试、本地实时监控面板、零依赖 GPU 检测与模型权重审计。

## 单次请求调试 (`aib req`)

::: code-group

```bash [文本 Prompt]
aib req \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --prompt "What is prefix caching?"
```

```bash [文件与管道 Stdin]
# 从 JSON 文件发送
aib req --payload-path request.json

# 从 stdin 读取 JSON 请求体（以下等价）
cat request.json | aib req --payload-path - --model your-model
cat request.json | aib req

# 从 stdin 读取纯文本 prompt
cat prompt.txt | aib req --prompt - --max-tokens 128
```

```bash [覆盖参数 / 思考模式]
aib req \
  --prompt "Solve x + 2 = 5" \
  --disable-thinking \
  --override-payload '{"temperature":0,"seed":42}'
```

:::

- 默认流式输出；支持 `--disable-stream`（单次 JSON）、`--raw`（原始 SSE）与 `--no-print`（仅打印指标）。含 `prompt` 且不含 `messages` 的请求自动路由至 `/v1/completions`。
- 进程退出码与请求成功与否绑定：`0` 表示收到完整正常响应，`1` 表示连接失败、非 200 状态、流异常中断或生成被中止（可用于探活脚本），`2` 表示 CLI 参数或输入文件错误。

## 本地监控 (`aib monitor`)

一键拉起本地 Prometheus（`:9090`）与 Grafana（`:3000`，内置 SGLang 看板），并在终端交互动态增删节点：

```bash
# 被测服务需启用 --enable-metrics
aib monitor --base-urls http://127.0.0.1:30000 --open
```

| 选项 | 说明 |
| --- | --- |
| `--runtime auto\|system\|download` | 自动复用系统二进制或下载经校验的 Prometheus / Grafana |
| `--no-grafana` | 仅启动 Prometheus |
| `--scrape-interval 5s` | 指标拉取间隔（默认 `5s`） |
| `--dry-run` | 仅生成配置不启动子进程 |

## 硬件、权重与辅助工具

::: code-group

```bash [GPU 与权重检查]
# 纯标准库 + NVIDIA 驱动直检显卡规格与拓扑
aib check-device --json

# 轻量统计 safetensors / .bin 模型参数量
aib check-weight --path /path/to/model
```

```bash [Tokenizer 编解码]
aib token --tokenizer /path/to/tokenizer "Hello world"
echo '[1, 2, 3]' | aib token --tokenizer /path/to/tokenizer --decode -
```

```bash [SGLang Profile & Token ID 导出]
aib profile-sgl --base-url http://127.0.0.1:30000 --num-steps 5
aib reply --base-url http://127.0.0.1:30000 --payload-path requests.jsonl -o reply.jsonl
```

:::
