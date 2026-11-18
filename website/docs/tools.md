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

## 跨集群实时监控 (`aib monitor`)

启动 Prometheus（`:9090`）、Grafana（`:3000`，默认开放 Admin 可视化编辑与持久化）及 HTTP 注册/上报中心（`:9091`），支持跨集群 Pod 启动自动注册与 NAT 隔离环境主动上报：

::: code-group

```bash [1. 启动监控服务端]
# 无需预先指定 --base-urls；后台运行自动进入无头模式，支持 --dashboard 导入自定义 JSON
aib monitor --listen-address 0.0.0.0 --allow-remote
```

```bash [2. 直连集群自动注册 (Pull)]
# Pod 启动钩子：使用 aib 或零依赖 curl 注册，Prometheus 通过 file_sd 自动热加载
aib monitor --register http://<monitor-ip>:9091 \
  --base-urls http://10.12.8.5:30000 --labels cluster=bj model=qwen3

curl -X POST http://<monitor-ip>:9091/api/register \
  -d '{"url":"http://10.12.8.5:30000","labels":{"cluster":"bj","model":"qwen3"}}'
```

```bash [3. NAT 隔离集群上报 (Push)]
# 中心无法反连 Pod IP 时，在 Pod 侧周期抓取本地 /metrics 并推送至中心（停机自动注销）
aib monitor --push-to http://<monitor-ip>:9091 \
  --base-urls http://127.0.0.1:30000 --labels cluster=sh-nat model=qwen3
```

:::

| 选项 | 说明 |
| --- | --- |
| `--registry-port 9091` | 跨集群 `/api/register`、`/api/deregister` 与 `/api/push` 监听端口 |
| `--labels KEY=VALUE ...` | 为目标附加维度标签（如 `cluster`、`model`，大盘内置对应下拉筛选） |
| `--dashboard PATH` | 导入自定义 Grafana JSON 文件或目录；浏览器内修改保存后重启不丢失 |
| `--runtime auto\|system\|download` | 自动复用系统二进制或下载经 SHA256 校验的 Prometheus / Grafana |
| `--headless` / `--no-grafana` | 无交互常驻运行（非 TTY 自动开启）/ 仅运行 Prometheus + Registry |

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
