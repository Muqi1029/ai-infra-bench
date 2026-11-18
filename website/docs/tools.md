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

## 实时监控与压测对比平台 (`aib monitor` / `aib studio`)

单命令在固定 CPU 服务器或开发机上启动 **Prometheus（`:9090`）+ Grafana（`:3000`）+ 跨集群注册中心与 Web 压测/正确性工作台（`:9091`）**：支持所有推理服务启动自动注册上线持续监控，并在网页端一键发起 `bench` / `slo` / `eval-dataset` / `eval-logits` / `eval-hidden-states`、固化窗口监控切片至 SQLite 进行跨版本对比。

::: code-group

```bash [1. 启动监控与压测平台]
# 部署在固定 IP 节点（后台自动进入无头模式，访问 http://<ip>:9091 打开 Studio Web 界面）
aib monitor --listen-address 0.0.0.0 --allow-remote
# 等价别名：aib studio --listen-address 0.0.0.0 --allow-remote
```

```bash [2. 零侵入包裹启动自动注册 (推荐)]
# 任意 GPU 节点启动推理服务时自动提取端口/模型/硬件规格，就绪后自动注册监控（直连 Pull 或跨 NAT Push），停机自动下线
aib monitor --server http://<monitor-ip>:9091 --labels cluster=bj -- \
  python3 -m sglang.launch_server --model-path /models/Qwen3-8B --port 30000 --enable-metrics

# 也可预设环境变量
export AIB_MONITOR_URL=http://<monitor-ip>:9091
aib monitor -- vllm serve /models/Qwen3-8B --port 8000
```

```bash [3. CLI 主动上报 / 独立注册]
# 终端执行压测或正确性校验并同步持久化至中心平台对比
aib bench --base-url http://127.0.0.1:30000 --dataset random --report-to http://<monitor-ip>:9091
aib eval-dataset --evals gsm8k --base-url http://127.0.0.1:30000 --report-to http://<monitor-ip>:9091

# Python 内嵌或已运行实例手动上报监控
python3 -c 'from ai_infra_bench.monitor import auto_register; auto_register("http://<monitor-ip>:9091", port=30000)'
curl -X POST http://<monitor-ip>:9091/api/register -d '{"url":"http://10.12.8.5:30000","labels":{"cluster":"bj"}}'
```

:::

| 选项 | 说明 |
| --- | --- |
| `--server URL -- <cmd>` | 包裹推理进程启动并自动完成就绪探测、硬件/参数上报、Pull/Push 协商注册与停机注销（支持 `$AIB_MONITOR_URL`） |
| `--registry-port` / `--port 9091` | Studio Web 界面、`/api/register`、`/api/push` 与压测对比 API 端口 |
| `--labels KEY=VALUE ...` | 为目标附加维度标签（如 `cluster`、`model`，支持 `$AIB_MONITOR_LABELS` 与大盘下拉筛选） |
| `--dashboard PATH` | 导入自定义 Grafana JSON 文件或目录；浏览器内修改保存后重启不丢失 |
| `--runtime auto\|system\|download` | 自动复用系统二进制或下载经 SHA256 校验的 Prometheus / Grafana |
| `--no-prometheus` / `--no-grafana` | 仅启动轻量 Web 工作台与注册中心（无需外部二进制）/ 仅关闭 Grafana |

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
