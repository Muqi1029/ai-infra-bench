# Requests & Tools

Single-request stdin debugging, local Prometheus + Grafana telemetry, zero-dependency GPU detection, and weight inspection.

## Request Debugging (`aib req`)

::: code-group

```bash [Prompt]
aib req \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --prompt "What is prefix caching?"
```

```bash [File & Stdin (-)]
# Send JSON from file
aib req --payload-path request.json

# Read JSON payload from stdin (equivalent forms)
cat request.json | aib req --payload-path - --model your-model
cat request.json | aib req

# Read plain-text prompt from stdin
cat prompt.txt | aib req --prompt - --max-tokens 128
```

```bash [Override & Thinking]
aib req \
  --prompt "Solve x + 2 = 5" \
  --disable-thinking \
  --override-payload '{"temperature":0,"seed":42}'
```

:::

- Streams by default; pass `--disable-stream` for single JSON, `--raw` for raw SSE frames, or `--no-print` for metrics only. Payloads with `prompt` and no `messages` route to `/v1/completions`.
- Process exit code reflects request health: `0` on a complete valid response, `1` on connection/HTTP/stream/abort failure (suitable for health probes), and `2` on invalid CLI arguments or input files.

## Cross-Cluster Monitoring (`aib monitor`)

Launches Prometheus (`:9090`), Grafana (`:3000`, unlocked with Admin UI editing & persistence), and an HTTP Registry/Relay (`:9091`) with zero-intrusion startup wrapping and automatic Pull-vs-NAT-Push negotiation:

::: code-group

```bash [1. Start Monitor Server]
# Zero initial --base-urls required; auto-enters headless mode under non-TTY daemons
aib monitor --listen-address 0.0.0.0 --allow-remote
```

```bash [2. Zero-Intrusion Wrapper (Recommended)]
# Wrap any serving command: auto-detects port & model, waits for /metrics readiness,
# auto-negotiates direct Pull vs NAT Push, and deregisters cleanly on SIGTERM
aib monitor --server http://<monitor-ip>:9091 --labels cluster=bj -- \
  python3 -m sglang.launch_server --model-path /models/Qwen3-8B --port 30000

# Or pre-configure the monitor URL via env var in K8s / Docker without extra flags
export AIB_MONITOR_URL=http://<monitor-ip>:9091
aib monitor -- vllm serve /models/Qwen3-8B --port 8000
```

```bash [3. Python Hook / Manual Sidecar]
# One-liner inside custom Python servers (background probe + Pull/Push fallback + atexit cleanup)
python3 -c 'from ai_infra_bench.monitor import auto_register; auto_register("http://<monitor-ip>:9091", port=30000)'

# Or manual registration / NAT sidecar for already-running endpoints
curl -X POST http://<monitor-ip>:9091/api/register -d '{"url":"http://10.12.8.5:30000","labels":{"cluster":"bj"}}'
aib monitor --push-to http://<monitor-ip>:9091 --base-urls http://127.0.0.1:30000 --labels cluster=sh-nat
```

:::

| Option | Description |
| --- | --- |
| `--server URL -- <cmd>` | Wrap a service command for automatic readiness wait, Pull/Push registration, and shutdown deregistration (or `$AIB_MONITOR_URL`) |
| `--registry-port 9091` | HTTP port serving `/api/register`, `/api/deregister`, and `/api/push` |
| `--labels KEY=VALUE ...` | Attach labels (`cluster`, `model`, etc.; also reads `$AIB_MONITOR_LABELS`) mapped to Grafana filters |
| `--dashboard PATH` | Provision a custom Grafana `.json` file or folder; UI edits persist across restarts |
| `--runtime auto\|system\|download` | Use system binaries or download SHA256-verified Prometheus / Grafana |
| `--headless` / `--no-grafana` | Daemon mode without REPL (auto-enabled on non-TTY) / Prometheus + Registry only |

## Hardware, Weights & Utilities

::: code-group

```bash [GPU & Weights]
# Stdlib-only GPU spec & NVML driver query
aib check-device --json

# Lightweight safetensors / .bin parameter counter
aib check-weight --path /path/to/model
```

```bash [Tokenizer]
aib token --tokenizer /path/to/tokenizer "Hello world"
echo '[1, 2, 3]' | aib token --tokenizer /path/to/tokenizer --decode -
```

```bash [SGLang Profile & Reply]
aib profile-sgl --base-url http://127.0.0.1:30000 --num-steps 5
aib reply --base-url http://127.0.0.1:30000 --payload-path requests.jsonl -o reply.jsonl
```

:::
