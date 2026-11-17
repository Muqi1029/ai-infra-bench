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

## Local Monitoring (`aib monitor`)

Start Prometheus (`:9090`) and Grafana (`:3000` with SGLang dashboard) locally and edit scrape targets interactively:

```bash
# Requires --enable-metrics on your serving instances
aib monitor --base-urls http://127.0.0.1:30000 --open
```

| Option | Description |
| --- | --- |
| `--runtime auto\|system\|download` | Use system binaries or download checksum-verified Prometheus / Grafana |
| `--no-grafana` | Run Prometheus only |
| `--scrape-interval 5s` | Scrape period (default `5s`) |
| `--dry-run` | Write configs without launching daemons |

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
