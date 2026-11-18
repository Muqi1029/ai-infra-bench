# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React 19 + TypeScript + Tailwind CSS + Radix UI primitives (shadcn component architecture) + Lucide vector icons + Recharts, authored in `studio-ui/` and bundled via Vite (`vite-plugin-singlefile`) into self-contained offline assets under `ai_infra_bench/monitor/assets/studio.html`.

## Users

AI infrastructure and LLM inference engineers deploying, tuning, and verifying GPU serving backends (SGLang, vLLM, and OpenAI-compatible engines) across internal clusters, often accessing the control plane over SSH local port forwarding from an internal CPU jump/monitoring host.

## Product Purpose

`AI Infra Bench Studio` (`aib monitor` / `aib studio` / `aib platform`) unifies continuous real-time service monitoring with on-demand performance benchmarking (`aib bench`, `aib slo`), numerical and dataset correctness verification (`eval-dataset`, `eval-logits`, `eval-hidden-states`), and persistent SQLite A/B session comparison on a single fixed-IP server.

## Positioning

Unlike standalone load generators that discard server state or Prometheus/Grafana stacks that cannot orchestrate benchmarks or inspect token-level logits drift, AIB Studio binds live `/metrics` windows, hardware/launch arguments, throughput-interactivity Pareto curves, `<think>`-stripped dataset answers, and top-k logprob diffs to each concrete `ServiceSession` lifecycle with zero cluster scheduler or external database dependencies.

## Operating Context

- Deployed on a fixed-IP internal CPU server (`aib studio --port 9091` or SSH-forwarded `:8765`).
- GPU inference nodes wrap startup commands with `aib monitor --server http://<ip>:9091 -- ...` for automatic Pull/Push registration and graceful offline archival on exit, or engineers register existing backend URLs manually.
- Engineers repeatedly iterate on quantization (`bf16`, `fp8`, `nvfp4`), parallelism (`tp`, `dp`, `ep`, `cp`), and runtime flags, comparing candidate sessions against a pinned baseline.

## Capabilities and Constraints

- **100% Offline Runtime**: The packaged Python wheel/install must serve the Studio UI with zero runtime Node.js requirement and zero external CDN requests (fonts, icons, scripts, and styles must work on air-gapped corporate networks).
- **Per-Backend Job Serialization**: Concurrent jobs targeting the same backend URL queue automatically to prevent benchmark cross-talk; live progress and logs stream over `/api/events` (SSE).
- **Preserved HTTP & SSE Contracts**: Must integrate cleanly with `/api/platform/info`, `/api/monitor/live`, `/api/register`, `/api/deregister`, `/api/probe`, `/api/jobs`, `/api/jobs/<id>/cancel`, `/api/sessions`, `/api/compare`, `/api/evals/<id>/cases`, and `/api/events`.
- **Workspace Ergonomics**: Compact left sidebar navigation, flat unnested main work surfaces, and right-side slide-over drawers (`Sheet`) for secondary/deep inspection (live job terminal logs, advanced benchmark knobs, CLI auto-registration snippet, and per-case reasoning / per-position logits diffs).

## Evidence on Hand

- Existing Python backend and API server in `ai_infra_bench/monitor/` (`registry_server.py`, `storage.py`, `runner.py`, `telemetry.py`, `launcher.py`).
- End-to-end test suite in `tests/monitor/test_platform.py` and `tests/monitor/test_monitoring.py` (asserts `"AI Infra Bench Studio"` in root HTML response).
- Architecture and deployment documentation in `/data/muqi/task/tutorial/aib-platform/`.

## Product Principles

1. **Task Clarity Over Chrome**: Surface active fleet health, primary benchmark controls, and A/B deltas immediately on flat surfaces; push deep logs, secondary flags, and case-level diffs into on-demand slide-over sheets.
2. **Offline Self-Sufficiency**: Every visual asset, icon, chart engine, and UI primitive ships inside the local build artifact so internal cluster deployments never degrade without internet access.
3. **Configuration-Bound Truth**: Always tie throughput, latency, KV cache peaks, and accuracy deltas back to the exact server launch diff between Baseline and Candidate sessions.
4. **Tactile Control Precision**: Every input, select, switch, and action state must be unambiguous at a glance under both bright office monitors and dark terminal-adjacent workflows.
