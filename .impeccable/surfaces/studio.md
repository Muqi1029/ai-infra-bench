# Surface Brief: AI Infra Bench Studio (`ai_infra_bench/monitor/assets/studio.html`)

- **Scope**: Whole single-page operational workbench (`studio-ui/` -> `ai_infra_bench/monitor/assets/studio.html`)
- **Visitor Mode**: `Operate` (`code-led`)
- **Audience & Scene**: LLM inference engineers operating GPU backends across internal clusters over SSH tunnels, switching rapidly between terminal sessions and the browser to verify throughput, KV cache pressure, and numerical alignment.
- **Primary Tasks**:
  1. Monitor auto-registered and manually added inference backends in real time (`tok/s`, `KV Cache %`, `Prefix Hit %`, `Running/Waiting` requests).
  2. Configure and launch serialized `bench`, `eval-dataset`, `eval-logits`, `eval-hidden-states`, and `slo` jobs without form clutter.
  3. Compare Candidate vs. pinned Baseline sessions across configuration diffs, Pareto frontiers, load-sweep curves, and windowed server KV peaks.
  4. Audit dataset accuracy, `length` truncation rates, `<think>`-stripped answers, and token-position Top-K logprobs.
- **Confirmed Structural Constraints**:
  - Compact left navigation rail + flat single-layer main workspace + right slide-over Sheets (Drawers) for secondary configuration, live job console logs, CLI onboarding commands, and case/logits diff inspection.
  - Zero nested cards, zero Unicode/emoji icon substitutes, 100% offline bundle.

## Direction contract

THESIS: AIB Studio operates as a precision metrology and telemetry bench rather than a generic SaaS card grid, eliminating nested boxes and modal interruptions in favor of hairline-divided flat work tables paired with deep right-hand inspection sheets.

OWN-WORLD: Anchored deep graphite-indigo navigation rail (`#111622`) against a cool mineral work ground (`#f3f5f8` light / `#0b0f17` dark) and crisp flat work planes (`#ffffff` / `#121826`) separated by 1px structural hairlines (`#dce2ec` / `#1f293d`); cobalt-cyan operational accent (`#0284c7` / `#38bdf8`), emerald/amber/crimson semantic states, crisp geometric UI sans paired with tabular monospace strictly for endpoints, parameters, and numeric readouts, and uniform 1.75px Lucide vector icons.

STORY: On arrival, the engineer immediately reads live cluster health and KV-cache load across registered backends, launches clean benchmark or numerical verification suites in three clicks, evaluates A/B session deltas on the Pareto frontier, and pulls open a slide-over sheet whenever raw logs, advanced sweep knobs, or per-token logprob diffs are needed.

FIRST VIEWPORT: A fixed `228px` graphite-indigo left rail (`AI Infra Bench Studio` title, live SSE pulse badge, four primary workspace nav items with live counts, active-job status trigger, refresh and theme controls) beside a full-height work canvas opening with a flat inline Backend URL registration bar and `接入命令` sheet button, followed by the signature Fleet Telemetry Ledger (structured rows with inline KV-cache fill bars, tabular `tok/s` & `Run/Wait` readouts, and row actions) above dual side-by-side Recharts (`实时生成吞吐` and `KV Cache & Prefix Hit`).

FORM: Precision Telemetry Bench (position 1, `seed: none (unavailable)`).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
