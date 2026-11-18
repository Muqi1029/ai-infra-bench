# AIB Studio UI (`studio-ui/`)

Frontend workspace for **AI Infra Bench Studio** (`aib monitor` / `aib studio` / `aib platform`), built with:

- **React 19 + TypeScript**
- **Radix UI** (`Select`, `Dialog` / Slide-over `Sheet`, `Switch`, `Tabs`, `Label`)
- **Tailwind CSS v4**
- **Lucide React** (1.75px vector icon system)
- **Recharts** (live telemetry curves, Pareto frontiers, and load-sweep charts)
- **Vite + `vite-plugin-singlefile`** (bundles JS and CSS into a 100% offline-capable single HTML asset served directly by `ai_infra_bench/monitor/registry_server.py`)

## Build

Requires Node.js >= 20 (for example `nvm use 22`):

```bash
cd studio-ui
npm install
npm run build
```

`npm run build` outputs directly to `../ai_infra_bench/monitor/assets/studio.html`. End users installing `ai-infra-bench` do not need Node.js or external internet access at runtime.
