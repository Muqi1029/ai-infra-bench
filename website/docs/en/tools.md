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

## Real-Time Monitoring & Benchmark Studio (`aib monitor` / `aib studio`)

Launches Prometheus (`:9090`), Grafana (`:3000`), and the unified Registry & Benchmark Studio Web UI (`:9091`). Inference backends auto-register on startup for continuous monitoring, while the Web UI runs `bench` / `slo` / `eval-dataset` / `eval-logits` / `eval-hidden-states` and persists windowed server telemetry into SQLite for side-by-side comparison:

::: code-group

```bash [1. Start Platform Server]
# Deploy on a fixed-IP CPU host (open http://<ip>:9091 for Studio Web UI)
aib monitor --listen-address 0.0.0.0 --allow-remote
# Alias: aib studio --listen-address 0.0.0.0 --allow-remote
```

```bash [2. Zero-Intrusion Wrapper (Recommended)]
# Wrap any serving command: auto-detects port/model/GPUs, waits for /metrics readiness,
# auto-negotiates direct Pull vs NAT Push, and deregisters cleanly on SIGTERM
aib monitor --server http://<monitor-ip>:9091 --labels cluster=bj -- \
  python3 -m sglang.launch_server --model-path /models/Qwen3-8B --port 30000 --enable-metrics

# Or pre-configure the monitor URL via env var
export AIB_MONITOR_URL=http://<monitor-ip>:9091
aib monitor -- vllm serve /models/Qwen3-8B --port 8000
```

```bash [3. CLI Ingestion / Manual Hook]
# Upload CLI benchmark or correctness runs directly to the central Studio ledger
aib bench --base-url http://127.0.0.1:30000 --dataset random --report-to http://<monitor-ip>:9091
aib eval-dataset --evals gsm8k --base-url http://127.0.0.1:30000 --report-to http://<monitor-ip>:9091

# Python hook or manual registration
python3 -c 'from ai_infra_bench.monitor import auto_register; auto_register("http://<monitor-ip>:9091", port=30000)'
curl -X POST http://<monitor-ip>:9091/api/register -d '{"url":"http://10.12.8.5:30000","labels":{"cluster":"bj"}}'
```

:::

| Option | Description |
| --- | --- |
| `--server URL -- <cmd>` | Wrap a service command for automatic readiness wait, metadata collection, Pull/Push registration, and shutdown deregistration (or `$AIB_MONITOR_URL`) |
| `--registry-port` / `--port 9091` | HTTP port serving Studio Web UI, `/api/register`, `/api/push`, and comparison APIs |
| `--labels KEY=VALUE ...` | Attach labels (`cluster`, `model`, etc.; also reads `$AIB_MONITOR_LABELS`) mapped to Grafana filters |
| `--dashboard PATH` | Provision a custom Grafana `.json` file or folder; UI edits persist across restarts |
| `--runtime auto\|system\|download` | Use system binaries or download SHA256-verified Prometheus / Grafana |
| `--no-prometheus` / `--no-grafana` | Run Studio Web UI + SQLite + Registry without Prometheus/Grafana binaries / disable Grafana only |

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
