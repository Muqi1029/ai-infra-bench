---
name: "AI Infra Bench Studio"
description: "Precision telemetry, load benchmarking, and numerical alignment workbench for GPU LLM inference clusters."
colors:
  cobalt-brand: "#0369a1"
  cyan-telemetry: "#38bdf8"
  deep-marine-ink: "#06283d"
  cobalt-tint: "rgb(3 105 161 / 0.10)"
  emerald-signal-light: "#047857"
  emerald-signal-dark: "#34d399"
  emerald-tint: "rgb(4 120 87 / 0.10)"
  amber-threshold-light: "#b45309"
  amber-threshold-dark: "#fbbf24"
  amber-tint: "rgb(180 83 9 / 0.12)"
  crimson-fault-light: "#dc2626"
  crimson-fault-dark: "#f87171"
  crimson-tint: "rgb(220 38 38 / 0.10)"
  rail-graphite: "#111622"
  rail-slate: "#1a2234"
  rail-hairline: "#232d42"
  mineral-ground: "#f3f5f8"
  obsidian-ground: "#0b0f17"
  crisp-plane: "#ffffff"
  midnight-plane: "#121826"
  recessed-well-dark: "#090d16"
  subtle-wash-light: "#eef2f7"
  subtle-wash-dark: "#182032"
  structural-hairline-light: "#dce2ec"
  structural-hairline-dark: "#212c42"
  tactile-control-stroke: "#64748b"
  slate-ink: "#0f172a"
  frost-ink: "#f1f5f9"
  graphite-secondary: "#334155"
  mist-secondary: "#cbd5e1"
  steel-muted: "#475569"
  slate-muted: "#94a3b8"
typography:
  display:
    fontFamily: "Geist Mono Variable, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.4
    fontFeature: "'tnum' 1, 'zero' 1"
  headline:
    fontFamily: "Geist Variable, -apple-system, BlinkMacSystemFont, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"
    fontSize: "1rem"
    fontWeight: 700
    lineHeight: 1.5
  title:
    fontFamily: "Geist Variable, -apple-system, BlinkMacSystemFont, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 700
    lineHeight: 1.43
  body:
    fontFamily: "Geist Variable, -apple-system, BlinkMacSystemFont, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Geist Variable, -apple-system, BlinkMacSystemFont, PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "0.05em"
  mono:
    fontFamily: "Geist Mono Variable, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    fontSize: "0.78125rem"
    fontWeight: 500
    lineHeight: 1.5
    fontFeature: "'tnum' 1, 'zero' 1"
rounded:
  xs: "6px"
  sm: "7px"
  md: "8px"
  lg: "10px"
  xl: "12px"
  pill: "9999px"
spacing:
  2xs: "2px"
  xs: "6px"
  sm: "10px"
  md: "12px"
  lg: "16px"
  xl: "20px"
  2xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.cobalt-brand}"
    textColor: "{colors.crisp-plane}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "38px"
  button-secondary:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "38px"
  button-secondary-hover:
    backgroundColor: "{colors.subtle-wash-light}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.graphite-secondary}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "38px"
  button-ghost-hover:
    backgroundColor: "{colors.subtle-wash-light}"
    textColor: "{colors.slate-ink}"
  button-danger:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.crimson-fault-light}"
    rounded: "{rounded.md}"
    padding: "6px 14px"
    height: "38px"
  button-danger-hover:
    backgroundColor: "{colors.crimson-tint}"
  input-field:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.slate-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "6px 12px"
    height: "38px"
  select-trigger:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.md}"
    padding: "6px 12px"
    height: "38px"
  tabs-list:
    backgroundColor: "{colors.subtle-wash-light}"
    textColor: "{colors.steel-muted}"
    rounded: "{rounded.md}"
    padding: "2px"
  tabs-trigger-active:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.xs}"
    padding: "4px 10px"
  badge-brand:
    backgroundColor: "{colors.cobalt-tint}"
    textColor: "{colors.cobalt-brand}"
    rounded: "{rounded.xs}"
    padding: "2px 8px"
  badge-success:
    backgroundColor: "{colors.emerald-tint}"
    textColor: "{colors.emerald-signal-light}"
    rounded: "{rounded.xs}"
    padding: "2px 8px"
  rail-nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.slate-muted}"
    rounded: "{rounded.md}"
    padding: "10px 12px"
  rail-nav-item-active:
    backgroundColor: "{colors.rail-slate}"
    textColor: "{colors.crisp-plane}"
    rounded: "{rounded.md}"
    padding: "10px 12px"
  work-plane:
    backgroundColor: "{colors.crisp-plane}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.xl}"
    padding: "20px"
---

# Design System: AI Infra Bench Studio

## Overview

**Creative North Star: "The Precision Telemetry Bench"**

AI Infra Bench Studio is an operational metrology and verification workbench built for LLM inference engineers tuning GPU serving backends across air-gapped clusters. Its visual identity trades generic SaaS card grids and decorative chrome for calm, high-density instrumentation: a permanently dark Graphite-Indigo navigation rail (`#111622`) anchors the viewport while the main workspace lays out flat, unnested work planes (`#ffffff` in light mode, `#121826` in dark mode) partitioned by crisp `1px` structural hairlines (`#dce2ec` light, `#212c42` dark). Every font binary, vector icon, and chart primitive ships self-contained inside the single-file offline bundle so corporate SSH-tunneled sessions never degrade without external network access.

Because engineers pivot rapidly between dark terminal windows and bright office monitors, the system pairs dual-scheme `light-dark()` surfaces with high-visibility `#64748b` tactile control boundaries and tabular slashed-zero monospace strictly reserved for endpoints, launch parameters, and live telemetry numbers. Where the initial direction brief proposed `#0284c7` for light-mode accents and `#1f293d` for dark hairlines, the shipped build deepens the light operational accent to `#0369a1` to guarantee ≥4.5:1 white-label contrast on primary actions and calibrates dark structural hairlines to `#212c42` for cleaner plane separation. Nested cards, eyebrow kickers, Unicode glyph substitutes, and modal interruptions for non-destructive inspection are rejected system-wide; deep logs, advanced sweep knobs, and token-level logprob diffs slide out on demand in right-hand drawers.

**Key Characteristics:**
- **Anchored Graphite-Indigo Rail:** A fixed `228px` dark navigation rail (`#111622`) frames live SSE connectivity, workspace counts, and active job progress identically across light and dark themes.
- **Flat Hairline-Divided Planes:** Unnested `12px`-radius work planes replace box-inside-box layouts with `1px` structural hairlines (`#dce2ec` / `#212c42`) across forms, multi-column A/B strips, and paired telemetry charts.
- **Tactile High-Contrast Controls:** Inputs, selects, switches, and secondary buttons enforce a dedicated `#64748b` stroke (≥3.9:1 boundary contrast) rather than fading into passive divider lines.
- **Disciplined Metrology Typography:** Variable geometric UI sans (`Geist Variable`) handles bilingual labels and prose while `Geist Mono Variable` (`tabular-nums slashed-zero`) locks numeric columns, endpoints, and logprob deltas against horizontal jitter.
- **Offline-Complete Craft:** Self-hosted variable WOFF2 fonts and uniform `1.75px`-stroke Lucide SVG icons deliver full visual fidelity on air-gapped GPU clusters.

## Colors

Cool mineral and obsidian work grounds pair with a high-contrast cobalt-cyan operational accent and calibrated three-stage telemetry signals for throughput, cache pressure, and numerical alignment.

### Primary
- **Operational Cobalt** (`#0369a1`): Primary interactive fill, active chart stroke, focus ring, and link accent in light mode (`--brand-light`), darkened from sky-blue so crisp white button labels (`#ffffff`) clear WCAG AA contrast.
- **Telemetry Sky** (`#38bdf8`): Primary interactive fill, active focus outline, and primary chart series in dark mode (`--brand-dark`), as well as the theme-invariant rail gauge badge and active rail indicator (`--rail-accent`).
- **Deep Marine Ink** (`#06283d`): High-contrast dark foreground ink (`--brand-fg-dark`) paired over `#38bdf8` primary buttons, checked switches, and the rail masthead icon tile.
- **Cobalt Wash** (`rgb(3 105 161 / 0.10)`): Translucent primary surface tint (`--brand-soft-light`, paired with `rgb(56 189 248 / 0.14)` in dark mode) for selected telemetry rows, active suite toggles, configuration diff badges, and text `::selection`.

### Secondary
- **Alignment Emerald** (`#047857`): Light-mode positive signal (`--emerald-light`) for `Pull` connection badges, `PASS` correctness verdicts, favorable A/B throughput/latency deltas, and secondary chart series (`Prefix Hit %` and candidate load sweeps).
- **Live Emerald** (`#34d399`): Dark-mode positive signal (`--emerald-dark`) and the live SSE connection pulse dot in the navigation rail.
- **Emerald Wash** (`rgb(4 120 87 / 0.10)`): Light-mode soft fill (`--emerald-soft-light`, paired with `rgb(52 211 153 / 0.14)` in dark mode) behind `success` status badges.

### Tertiary
- **Threshold Ochre** (`#b45309`): Light-mode warning ink (`--amber-light`) for `Push` reporting badges, `length` truncation warnings, and KV-cache warning fills above `68%`.
- **Saturation Amber** (`#fbbf24`): Dark-mode warning signal (`--amber-dark`), `85%` KV-cache saturation reference lines on live telemetry charts, and baseline KV peak comparison curves.
- **Amber Wash** (`rgb(180 83 9 / 0.12)`): Light-mode soft fill (`--amber-soft-light`, paired with `rgb(251 191 36 / 0.15)` in dark mode) behind `warning` badges.
- **Fault Crimson** (`#dc2626`): Light-mode critical signal (`--crimson-light`) for KV-cache saturation fills above `85%`, `FAIL` alignment verdicts, regressed A/B deltas, `:user-invalid` field borders, and destructive confirmation buttons.
- **Alert Coral** (`#f87171`): Dark-mode critical signal (`--crimson-dark`) for regression deltas, `85%` KV peak threshold lines, and destructive outline controls on dark surfaces.
- **Crimson Wash** (`rgb(220 38 38 / 0.10)`): Light-mode soft fill (`--crimson-soft-light`, paired with `rgb(248 113 113 / 0.15)` in dark mode) for `danger` badges and destructive button hover states.

### Neutral
- **Rail Graphite-Indigo** (`#111622`): Theme-invariant background (`--rail-bg`) for the desktop left navigation rail and mobile sticky header.
- **Rail Active Slate** (`#1a2234`): Elevated rail surface (`--rail-surface`) for active workspace navigation items and hovered rail footer controls.
- **Rail Hairline** (`#232d42`): Structural `1px` border (`--rail-border`) dividing the rail header, footer, and utility buttons.
- **Cool Mineral Ground** (`#f3f5f8`): Light-mode page canvas (`--bg-light`) creating crisp figure-ground contrast beneath white work planes without drop shadows.
- **Obsidian Ground** (`#0b0f17`): Dark-mode page canvas (`--bg-dark`) tuned for low-glare parity next to dark SSH terminal windows.
- **Crisp Work Plane** (`#ffffff`): Light-mode primary container plane (`--surface-light`) and interactive field background (`--control-bg-light`).
- **Midnight Work Plane** (`#121826`): Dark-mode primary container plane (`--surface-dark`) for tables, forms, chart panels, and slide-over drawers.
- **Recessed Control Well** (`#090d16`): Deepened dark-mode field background (`--control-bg-dark`) that visually recesses inputs, selects, and log consoles inside `#121826` planes.
- **Subtle Mineral Wash** (`#eef2f7`): Light-mode secondary fill (`--surface-subtle-light`) for table `<thead>` bands, segmented `TabsList` tracks, row hovers, and config-diff strips.
- **Subtle Slate Wash** (`#182032`): Dark-mode secondary fill (`--surface-subtle-dark`) for table headers, segmented tab tracks, and row hover states.
- **Light Structural Hairline** (`#dce2ec`): Light-mode `1px` border (`--border-light`) for outer plane perimeters and internal `divide-x` / `divide-y` partitions.
- **Dark Structural Hairline** (`#212c42`): Dark-mode `1px` border (`--border-dark`) separating planes, table rows, and drawer headers.
- **Tactile Control Stroke** (`#64748b`): Shared light/dark `1px` border (`--control-border`) on inputs, selects, switches, secondary buttons, and dashed empty states.
- **Deep Slate Ink** (`#0f172a`): Primary light-mode text (`--text-light`) and dark scrollbar track (`--scroll-track-dark`).
- **Frost Ink** (`#f1f5f9`): Primary dark-mode text (`--text-dark`) and primary navigation rail text (`--rail-text`).
- **Graphite Secondary Ink** (`#334155`): Secondary light-mode text (`--text-sec-light`) for form labels and subheads, doubling as the light-mode control hover border (`--control-border-hover-light`).
- **Mist Secondary Ink** (`#cbd5e1`): Secondary dark-mode text (`--text-sec-dark`) for form labels and table metadata.
- **Steel Muted Ink** (`#475569`): Muted light-mode text (`--text-muted-light`) for helper descriptions, placeholders, axis ticks, and uppercase table column headers.
- **Muted Slate Ink** (`#94a3b8`): Muted dark-mode text (`--text-muted-dark`), inactive rail text (`--rail-muted`), offline status dot, and dark-mode control hover border (`--control-border-hover-dark`).

### Named Rules
**The Anchored Rail Rule.** The left navigation rail remains locked to Graphite-Indigo (`#111622`) with `#232d42` borders and `#38bdf8` active accents in both light and dark modes, preserving a constant instrument-frame anchor while main work planes adapt to ambient lighting.

**The Tactile Boundary Rule.** Interactive controls (`Input`, `SelectTrigger`, `Switch`, and secondary `Button`) must use Tactile Control Stroke (`#64748b`) rather than passive structural hairlines (`#dce2ec` or `#212c42`), sharpening to `#334155` (light) or `#94a3b8` (dark) on hover so hit targets remain unmistakable.

**The Calibrated Signal Rule.** Emerald, Amber, and Crimson appear strictly to communicate live connection mode, KV-cache saturation (`>68%` Amber, `>85%` Crimson), correctness verdicts, A/B regressions, or destructive actions—never as decorative accents.

## Typography

**Display Font:** Geist Mono Variable (with `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace` fallback)
**Body Font:** Geist Variable (with `-apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif` fallback)
**Label/Mono Font:** Geist Mono Variable (self-hosted variable WOFF2 with `font-variant-numeric: tabular-nums slashed-zero`)

**Character:** Crisp geometric variable sans pairs seamlessly with CJK system fallbacks for dense bilingual engineering controls, while tabular slashed-zero monospace gives URLs, CLI flags, and streaming telemetry metrics mechanical alignment.

### Hierarchy
- **Display** (700, `1.25rem` / `20px`, line-height `1.4`): Primary numerical instrument readouts (`tok/s` peak output throughput and `ms` minimum TPOT) inside session A/B comparison strips.
- **Headline** (700, `1rem` / `16px`, line-height `1.5`): Top-level workspace view headings, slide-over `SheetTitle` headers, and `ConfirmDialog` titles (paired with `0.9375rem` / `15px` empty-state headings).
- **Title** (700, `0.875rem` / `14px`, line-height `1.43`): Flat panel headers, telemetry ledger titles, chart headers, form section titles (`13px`–`14px`), and the `13.5px` (`-0.025em` tracking) rail masthead.
- **Body** (400, `0.875rem` / `14px`, line-height `1.5`): Root document text; compact operational controls and ledger cells step down to `13px`–`13.5px` (`600` weight on buttons), muted helper descriptions sit at `12px`, and centered empty-state copy caps measure at `max-w-lg` to `max-w-xl` (`512px`–`576px`, ~55–65ch).
- **Label** (600, `0.6875rem` / `11px`, letter-spacing `0.05em`, uppercase): Dense data-table `<thead>` column headers (`11px`–`11.5px`), complemented by `12.5px` sentence-case (`600` weight, normal tracking) form `<Label>` elements above inputs.
- **Mono** (500, `0.78125rem` / `12.5px`, line-height `1.5`, `'tnum' 1, 'zero' 1`): Backend URLs, model identifiers, launch parameters, CLI snippets, job logs (`12px`), and per-position Top-K logprob tables.

### Named Rules
**The Strict Monospace Mandate Rule.** `Geist Mono Variable` with `tabular-nums slashed-zero` is reserved exclusively for URLs, CLI commands, parameter keys, job/session IDs, and numerical measurements—never applied to prose, headings, or navigation labels as a "technical" costume.

**The Tabular Ledger Rule.** Every data table (`tabular-nums`) and numeric readout (`tabular-mono`) locks character advance widths so 1-second SSE telemetry ticks never shift adjacent columns horizontally.

## Layout

The workbench uses a two-zone operational shell without marketing heroes or nested wrappers:
- **Desktop App Shell (`≥1024px`):** A fixed `228px` sticky full-viewport left rail (`h-screen sticky top-0`) sits beside a flexible main work canvas (`flex-1 min-w-0 max-w-[1480px]`) padded at `24px` (`16px` below `640px`).
- **Compact Viewport Shell (`<1024px`):** The fixed rail collapses into a sticky `52px` (`h-13`) top bar in `#111622` with a menu button that opens a `250px` left slide-over `Sheet` drawer, keeping live SSE status and job-console access one tap away.
- **Spacing Rhythm & Hairline Partitioning:** Major stacked work planes separate by `20px` (`gap-5`). Inside a work plane, related sub-sections—header bars, inline registration forms, 4-column A/B instrument strips, and side-by-side chart pairs—share one `12px`-radius outer container divided by `1px` `--border` hairlines (`divide-y` / `divide-x`), using `16px` (`p-4`) padding in chart/metric cells and `20px` (`p-5`) padding in form zones and drawer bodies. Form field groups use `6px` (`gap-1.5`) between label and control and `12px`–`16px` (`gap-3` / `gap-4`) between grid columns.
- **Responsive Grid Breakpoints:** `sm` (`640px`, 2-column A/B strip and `24px` canvas padding), `md` (`768px`, 2-column form grids, 3-column inline URL registration bar, and sticky right table action column), `lg` (`1024px`, persistent `228px` left rail and 2-column chart splits including the `0.88fr / 1.12fr` KV-peak + session-ledger split), and `xl` (`1280px`, 4-column A/B strip and 5-column suite switch row).
- **Slide-Over Progressive Disclosure & Motion:** Secondary tasks stay off the main plane inside right-hand `Sheet` drawers sized to their content (`448px` advanced bench knobs, `512px` CLI onboarding, `576px` live job console, `672px` dataset/logits diff inspector). Motion uses exponential ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`) at `220ms` for sheet entrances (`160ms` `cubic-bezier(0.4, 0, 1, 1)` on close), `180ms` for overlay fades and dialog pop-ins, and `2.4s` for the live SSE emerald pulse—all gated by `@media (prefers-reduced-motion: no-preference)`. Streaming Recharts series disable path animation (`isAnimationActive={false}`) for instant readout fidelity.

## Elevation & Depth

AI Infra Bench Studio uses a flat-by-default tonal and hairline architecture on the workspace canvas, reserving cast shadows exclusively for transient floating layers that interrupt or overlay the plane. At rest, depth comes from the contrast step between the Cool Mineral (`#f3f5f8`) or Obsidian (`#0b0f17`) ground, Crisp (`#ffffff`) or Midnight (`#121826`) work planes, and `#090d16` dark recessed control wells.

### Shadow Vocabulary
- **Overlay Shadow — Light** (`box-shadow: 0 14px 34px -8px rgb(15 23 42 / 0.22), 0 4px 10px -3px rgb(15 23 42 / 0.10)`): Assigned to `--overlay-shadow` in light mode exclusively for floating `SelectContent` menus, slide-over `Sheet` drawers, and `ConfirmDialog` modals.
- **Overlay Shadow — Dark** (`box-shadow: 0 18px 42px -10px rgb(0 0 0 / 0.65)`): Assigned to `--overlay-shadow` in dark mode so floating menus, drawers, and dialogs separate cleanly from `#0b0f17` and `#121826` surfaces.
- **Sticky Column Divider** (`box-shadow: -1px 0 0 var(--border)`): Structural `1px` hairline cast leftward by sticky right-hand table action cells (`≥768px`) during horizontal table scrolling.

### Named Rules
**The Flat Work Plane Rule.** Resting work planes, telemetry tables, A/B metric strips, and chart containers carry zero drop shadow (`box-shadow: none`); shadows appear only on portaled overlays (`SelectContent`, `SheetContent`, and `ConfirmDialog`).

**The Single Elevation Declaration Rule.** Resting containers declare their boundary once via a `1px` structural hairline (`#dce2ec` / `#212c42`) and surface-to-ground contrast—never pairing a card border with an ambient drop shadow.

## Shapes

Corner geometry steps tighter as component scale decreases, maintaining crisp concentric alignment without soft consumer rounding:
- **Outer Work Planes & Dialogs (`12px`):** All top-level flat panels, telemetry ledgers, multi-zone forms, and destructive `ConfirmDialog` modals use `12px` (`rounded.xl`) corners with `1px solid` `--border` strokes (or `1px dashed #64748b` for zero-data empty states).
- **Floating Menus & Terminal Wells (`10px`):** Portaled `SelectContent` poppers, dark log-output `<pre>` regions, empty-state icon tiles, and structured sub-panels inside drawers use `10px` (`rounded.lg`).
- **Standard Controls & Rail Items (`8px`):** Default `38px`-tall buttons, text inputs, select triggers, segmented `TabsList` containers, suite toggle rows, and rail navigation buttons use `8px` (`rounded.md`).
- **Compact Controls & Filter Chips (`7px`):** Small `32px` buttons, rail footer utility buttons, drawer close buttons, dataset selector chips, and token-position buttons use `7px` (`rounded.sm`).
- **Inline Badges & Active Segments (`6px`):** Semantic `Badge` tags, `TabsTrigger` pills inside `2px`-padded `TabsList` tracks, `SelectItem` rows, and inline `<code>` callouts use `6px` (`rounded.xs`).
- **Pills & Gauges (`9999px`):** Full pill rounding (`rounded.pill`) is restricted to `Switch` tracks and thumbs (`20px × 36px`), inline KV-cache and job progress bars (`6px`–`8px` tall), rail navigation count badges, and `6px` live SSE status dots—never applied to buttons or inputs.
- **Iconography:** All icons are vector SVGs from Lucide drawn at a uniform `1.75px` stroke width (`12px`–`16px` inside controls and badges, `20px` in empty-state tiles), stepping up to `2px`–`2.2px` only for micro selection checkmarks and the rail masthead gauge mark.

## Components

Every primitive favors crisp tactile boundaries, immediate state feedback (`150ms` color transitions), and unambiguous keyboard focus (`2px solid var(--brand)` at `2px` `outline-offset`).

### Buttons
Unambiguous operational triggers with `600` font weight and five semantic variants across four compact heights.
- **Shape:** `8px` radius on `md` (`38px` min-height, `6px 14px` padding, `13px` text), `lg` (`42px` min-height, `8px 18px` padding, `14px` text), and `icon` (`34px × 34px`); `7px` radius on `sm` (`32px` min-height, `4px 10px` padding, `12px` text).
- **Primary:** Operational Cobalt (`#0369a1`) with white text (`#ffffff`) in light mode; Telemetry Sky (`#38bdf8`) with Deep Marine Ink (`#06283d`) in dark mode, `1px` matching border.
- **Hover / Focus:** Secondary buttons shift stroke from `#64748b` to `#334155` (light) / `#94a3b8` (dark) over Subtle Wash (`#eef2f7` / `#182032`); danger outline buttons shift border to `--crimson` over `--crimson-soft`; `:focus-visible` renders a `2px` `--brand` outline with `2px` offset; `loading` prepends a `14px` spinning `Loader2` icon and disables pointer interaction at `50%` opacity.
- **Secondary / Ghost / Danger:** `secondary` uses `--control-bg` with `#64748b` stroke for toolbar and row actions; `ghost` sits transparent with `--text-sec` ink for low-weight links; `danger` pairs `--control-bg` and `#64748b` stroke with `--crimson` text for row deregister/delete triggers, escalating to `danger-solid` (`--crimson` fill, white text) only inside confirmation dialogs.

### Chips
Compact semantic `Badge` tags for telemetry state and bordered toggle chips for dataset and token-position filtering.
- **Style:** `Badge` uses `6px` radius, `2px 8px` padding, `11.5px` `600`-weight text, and borderless soft-wash fills across five tones (`neutral`, `brand`, `success`, `warning`, `danger`). Interactive filter chips use `7px` radius, `1px` stroke, and `tabular-mono` `12px` labels.
- **State:** Unselected filter chips rest on `--control-bg` with `#64748b` border and `--text-sec` text; selected (`aria-pressed="true"`) chips switch to `--brand` border, `--brand-soft` fill, and `--brand` text.

### Cards / Containers
Flat unnested work planes on the main canvas paired with portaled right-hand `Sheet` drawers for deep inspection.
- **Corner Style:** `12px` radius on main work planes and `ConfirmDialog`; `0px` full-height flush edge on `SheetContent` drawers.
- **Background:** Crisp Work Plane (`#ffffff`) in light mode and Midnight Work Plane (`#121826`) in dark mode; drawers dim the background with `rgb(0 0 0 / 0.50)` and `1.5px` backdrop blur.
- **Shadow Strategy:** `none` on all main work planes; `--overlay-shadow` exclusively on `SheetContent` and `ConfirmDialog` (see Elevation & Depth).
- **Border:** `1px solid` `--border` (`#dce2ec` / `#212c42`) around planes and between internal `divide-y` / `divide-x` zones; `1px dashed #64748b` on instructive empty states.
- **Internal Padding:** `14px 20px` (`px-5 py-3.5`) on plane headers, `16px` (`p-4`) on chart and metric cells, and `20px` (`p-5`) on form zones and scrollable sheet bodies.

### Inputs / Fields
High-clarity `38px` form fields, portaled selects, pill switches, and segmented tabs designed for rapid parameter entry.
- **Style:** `Input` and `SelectTrigger` share `38px` min-height, `8px` radius, `6px 12px` padding, `13.5px` text (`tabular-mono` on URLs, models, and numeric sweeps), `--control-bg` (`#ffffff` light, `#090d16` dark), and a `1px solid #64748b` border. `Switch` uses a `20px × 36px` (`9999px`) track; `TabsList` uses an `8px` `--surface-subtle` container with `2px` padding around `6px`-radius active triggers.
- **Focus:** `2px solid var(--brand)` outline at `2px` offset on `:focus-visible`; hover sharpens control borders to `#334155` (light) or `#94a3b8` (dark). Checked `SelectItem` rows highlight with `--brand-soft` fill, `600` weight, and a `14px` `--brand` checkmark.
- **Error / Disabled:** `:user-invalid` inputs switch border color to `--crimson`; disabled inputs and triggers drop to `55%` opacity (`50%` on switches and buttons) with `not-allowed` cursor.

### Navigation
A fixed `228px` Graphite-Indigo (`#111622`) left rail organized into brand masthead, four workspace triggers, and an operational footer.
- **Style & Typography:** Masthead pairs a `28px × 28px` (`7px` radius) `#38bdf8` gauge badge with `13.5px` bold title and `11px` `tabular-mono` version + live SSE pulse indicator. Nav items use `13px` `600`-weight labels with `16px` Lucide icons (`1.75px` stroke) and `9999px` tabular count pills.
- **Default / Hover / Active States:** Inactive items sit transparent in `#94a3b8` and hover to `rgb(26 34 52 / 0.60)` with `#ffffff` text. The active item (`aria-current="page"`) fills with `#1a2234`, turns text `#ffffff` and icon `#38bdf8`, shifts the count pill to `rgb(56 189 248 / 0.20)`, and pins a `3px`-wide `#38bdf8` vertical bar along its left edge.
- **Footer & Mobile Treatment:** Rail footer houses a live SSE job-progress card (opening the right-hand console sheet) above a 2-column `34px` Refresh + Theme toggle grid. Below `1024px`, navigation moves into a `52px` sticky `#111622` header with a `250px` left `Sheet` drawer.

### Fleet Telemetry Ledger
The signature operational matrix in `MonitorView` unifying live cluster health, KV-cache saturation, and chart focus selection in a single flat plane.
- **Structure:** A `12px`-radius plane opening with a `14px` bold header (`在线服务遥测矩阵`), horizontally scrollable data table (`12.5px` `tabular-nums`), `11px` uppercase column headers on `--surface-subtle`, `1px` `--border` row dividers, and a sticky right action column (`≥768px`) casting `-1px 0 0 var(--border)`.
- **Inline Telemetry & Thresholds:** Each row pairs `tabular-mono` instance/base-URL identity and `Cpu` hardware summary with `Pull`/`Push` mode badges, bold `tok/s` throughput, `Run/Wait` counts, and a `56px × 8px` (`9999px`) KV-cache fill bar that steps automatically from `--brand` (`≤68%`) to `--amber` (`>68%`) to `--crimson` (`>85%`). Clicking any row highlights it with `--brand-soft` and binds the dual Recharts panels below to that instance's 60-second SSE window.

## Do's and Don'ts

### Do:
- **Do** partition multi-part sections (registration bars, 4-column A/B strips, paired charts, and multi-zone forms) inside a single `12px`-radius `--surface` plane using `1px` `--border` hairlines (`divide-x` / `divide-y`).
- **Do** apply `Geist Mono Variable` (`tabular-nums slashed-zero`) to every URL, CLI flag, model/session slug, concurrency list, throughput figure, and logprob diff so live SSE updates never shift layout width.
- **Do** enforce the `#64748b` tactile border token (`--control-border`) on all text inputs, select triggers, switches, and secondary buttons, paired with a `2px` `--brand` `:focus-visible` outline at `2px` offset.
- **Do** route secondary workflows—live job terminal logs, advanced `aib bench` sweep parameters, CLI onboarding snippets, and per-case reasoning/logits diffs—into right-hand `Sheet` drawers, reserving centered `ConfirmDialog` modals strictly for destructive actions.
- **Do** theme native browser surfaces (`color-scheme`, `accent-color`, `caret-color`, `::selection`, and `scrollbar-color`) from the semantic palette and bundle all fonts and `1.75px` Lucide icons locally for 100% offline operation.

### Don't:
- **Don't** nest bordered or shadowed cards inside another card or work plane; use `1px` structural hairlines or `--surface-subtle` header/footer bands instead.
- **Don't** place uppercase eyebrow or kicker labels above section or panel headings—let the `14px`–`16px` bold heading and `12px` muted description carry hierarchy directly, keeping uppercase `11px` labels strictly inside table `<thead>` columns.
- **Don't** apply drop shadows to resting tables, metric strips, or form containers, and never combine a soft ambient shadow with a card border on resting surfaces.
- **Don't** use Unicode symbols or emoji as icon substitutes, or apply monospace to ordinary prose, button labels, or navigation items.
- **Don't** use pill (`9999px`) corners on buttons, inputs, or cards; keep interactive controls at `7px`–`8px` radius and reserve pills for switches, progress bars, count badges, and status dots.
