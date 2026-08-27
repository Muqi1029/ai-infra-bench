import html
import json
import math
import random
import re
from argparse import SUPPRESS, ArgumentParser
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class Color(Enum):
    LIGHT_CYAN = auto()
    LIGHT_GREEN = auto()
    LIGHT_YELLOW = auto()
    RED = auto()


def color_print(text: str, color: Color):
    RESET_CODE = "\033[0m"
    COLOR_TO_ANSI = {
        Color.LIGHT_CYAN: "\033[96m",
        Color.LIGHT_GREEN: "\033[92m",
        Color.LIGHT_YELLOW: "\033[93m",
        Color.RED: "\033[41m",
    }

    try:
        color_code = COLOR_TO_ANSI[color]
    except KeyError:
        raise NotImplementedError(f"{color} is not supported yet.")

    print(f"{color_code}{text}{RESET_CODE}", end="", flush=True)


def print_table(title: str, rows: List[List[str]]) -> None:
    print()
    if not rows:
        return

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    title_line = f"| {title.center(len(border) - 4)} |"

    print(border)
    print(title_line)
    print(border)
    for idx, row in enumerate(rows):
        print(
            "| "
            + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
            + " |"
        )
        if idx == 0:
            print(border)
    print(border)
    print()


def fmt(value, fmt: str = ".2f", suffix: str = "") -> str:
    """Format ``value`` with a printf-style format spec, e.g. ``fmt(1.23, ".2f")``."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value}{suffix}"
    if isinstance(value, float):
        return f"{value:{fmt}}{suffix}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_histogram_percentages(histogram: Sequence[int]) -> str:
    total = sum(histogram)
    if total == 0:
        return "[]"
    percentages = (f"{count / total:.2%}" for count in histogram)
    return f"[{', '.join(percentages)}]"


def format_mean(values: List[float], precision: int = 2) -> str:
    if not values:
        return "N/A"
    return f"{np.mean(values):.{precision}f}"


def format_percentile(
    values: List[float], percentile: float, precision: int = 2
) -> str:
    if not values:
        return "N/A"
    return f"{np.percentile(values, percentile):.{precision}f}"


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LEGACY_LABEL_PATTERN = re.compile(
    r"^(?P<label>.*)\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\)$"
)
_PAREN_PERCENT_PATTERN = re.compile(r"\(([-+]?\d+(?:\.\d+)?)%\)")


def _parse_plot_value(value: Any) -> float | List[float] | None:
    """Parse values emitted by ``handle_outputs`` into Plotly-friendly numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, list):
        parsed = [_parse_plot_value(item) for item in value]
        if all(isinstance(item, (int, float)) for item in parsed):
            return [float(item) for item in parsed]
        return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "UNLIMITED"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = [
                item.strip() for item in text[1:-1].split(",") if item.strip()
            ]
        return _parse_plot_value(parsed_json)

    match = _NUMBER_PATTERN.search(text)
    return float(match.group(0)) if match else None


def _validate_metric_records(
    records: Any, source_name: str = "metrics source"
) -> List[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return [records]
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        invalid_index = next(
            (
                index
                for index, record in enumerate(records, start=1)
                if not isinstance(record, Mapping)
            ),
            None,
        )
        if invalid_index is not None:
            raise ValueError(
                f"{source_name}: record {invalid_index} must be a JSON object"
            )
        return list(records)
    raise TypeError("metrics source must be a path, mapping, or sequence of mappings")


def _load_metric_records(source: Any) -> List[Mapping[str, Any]]:
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if source_path.suffix.lower() == ".jsonl":
            records = []
            with source_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{source_path}:{line_number}: invalid JSON ({error.msg})"
                        ) from error
                    if not isinstance(record, Mapping):
                        raise ValueError(
                            f"{source_path}:{line_number}: record must be a JSON object"
                        )
                    records.append(record)
            return records
        with source_path.open("r", encoding="utf-8") as stream:
            try:
                loaded = json.load(stream)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source_path}: invalid JSON ({error.msg})"
                ) from error
        return _validate_metric_records(loaded, str(source_path))

    return _validate_metric_records(source)


def _metric_sections(table_title: str, metric_name: str) -> List[str]:
    """Map a dumped metric to one or more dashboard sections."""
    table = table_title.lower()
    metric = metric_name.lower()
    sections: List[str] = []
    is_summary = "benchmark summary" in table
    if is_summary and metric in {
        "duration",
        "output throughput",
        "mean finished requests per second",
    }:
        sections.append("Overview")
    if "finish reason" in table:
        sections.append("Finish Reason")
    elif "spec" in table:
        sections.append("Spec")
    if "cache" in metric:
        sections.append("Cache")
    elif any(
        token in metric
        for token in ("prompt tokens", "completion tokens", "reasoning tokens")
    ):
        sections.append("Token Usage")
    elif any(token in metric for token in ("ttft", "tpot", "latency")):
        sections.append("Latency")
    if is_summary and not sections:
        return []
    return list(dict.fromkeys(sections or ["Other"]))


def _metric_specs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    labels: List[str] = []
    categories: Dict[str, Dict[str, Any]] = {}

    def add_point(
        category: str,
        metric_name: str,
        unit: str,
        label: str,
        concurrency: float,
        value: float,
        sections: Sequence[str],
    ) -> None:
        category_entry = categories.setdefault(
            category, {"name": category, "metrics": {}}
        )
        metric_entry = category_entry["metrics"].setdefault(
            metric_name,
            {
                "name": metric_name,
                "unit": unit,
                "section": sections[0],
                "sections": list(sections),
                "samples": {},
            },
        )
        label_samples = metric_entry["samples"].setdefault(label, {})
        label_samples.setdefault(concurrency, []).append(value)

    for record_index, record in enumerate(records, start=1):
        label = str(record.get("label") or record.get("Label") or f"run-{record_index}")
        if legacy_match := _LEGACY_LABEL_PATTERN.fullmatch(label):
            label = legacy_match.group("label") or "benchmark"
        if label not in labels:
            labels.append(label)

        concurrency_value = _parse_plot_value(record.get("max_concurrency"))
        if not isinstance(concurrency_value, float):
            summary_rows = record.get("Benchmark Summary") or []
            concurrency_value = next(
                (
                    parsed
                    for row in summary_rows
                    if isinstance(row, Mapping)
                    and str(row.get("Metric", "")).lower() == "max concurrency"
                    and isinstance(
                        (parsed := _parse_plot_value(row.get("Value"))), float
                    )
                ),
                float(record_index),
            )

        for table_title, rows in record.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                row_label_key = "Metric" if "Metric" in row else "Finish reason"
                row_label = row.get(row_label_key)
                if row_label is None:
                    continue
                for column, raw_value in row.items():
                    if column in {row_label_key, "Unit"}:
                        continue
                    value = _parse_plot_value(raw_value)
                    if value is None:
                        continue
                    suffix = "" if str(column).lower() == "value" else f" / {column}"
                    metric_name = f"{row_label}{suffix}"
                    unit = _plot_unit(raw_value, str(row.get("Unit") or ""))
                    sections = _metric_sections(str(table_title), str(row_label))
                    if not sections:
                        continue
                    if isinstance(value, list):
                        for bin_index, bin_value in enumerate(value):
                            add_point(
                                str(table_title),
                                f"{metric_name} / bin {bin_index}",
                                unit,
                                label,
                                concurrency_value,
                                bin_value,
                                sections,
                            )
                    else:
                        add_point(
                            str(table_title),
                            metric_name,
                            unit,
                            label,
                            concurrency_value,
                            value,
                            sections,
                        )
                        if ratio_match := _PAREN_PERCENT_PATTERN.search(str(raw_value)):
                            add_point(
                                str(table_title),
                                f"{metric_name} / Ratio",
                                "%",
                                label,
                                concurrency_value,
                                float(ratio_match.group(1)),
                                sections,
                            )

    category_order = [
        "Benchmark Summary",
        "Latency & Token Metrics",
        "Spec Tokens Statistics",
        "Finish Reason Statistics",
    ]
    category_order.extend(
        category_name
        for category_name in categories
        if category_name not in category_order
    )
    normalized_categories = []
    for category_name in category_order:
        category = categories.get(category_name)
        if category is None:
            continue
        metrics = []
        for metric in category["metrics"].values():
            series = {}
            for label, samples in metric.pop("samples").items():
                series[label] = [
                    {"x": x, "y": sum(values) / len(values)}
                    for x, values in sorted(samples.items())
                ]
            metrics.append({**metric, "series": series})
        normalized_categories.append({"name": category["name"], "metrics": metrics})

    return labels, normalized_categories


def _plot_unit(raw_value: Any, explicit_unit: str) -> str:
    text = str(raw_value).lower()
    if "%" in text and "tokens" not in text:
        return "%"
    for unit in ("tokens/s", "req/s", "ms", "tokens", "s"):
        if unit in text:
            return unit
    if "%" in text:
        return "%"
    return explicit_unit


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def export_metric_tables_html(
    metrics_source: Any,
    output_path: str | Path,
    *,
    sample_size: int | None = None,
    seed: int | None = None,
    title: str = "AI Infra Bench Metrics",
) -> Path:
    """Export dumped metric tables as a Grafana-style Plotly dashboard.

    ``metrics_source`` accepts a metrics JSON/JSONL path, one dumped mapping, or
    a sequence of dumped mappings. The first label is selected initially; the
    optional legacy ``sample_size`` argument remains available for callers that
    still want seeded random selection.
    """
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    records = _load_metric_records(metrics_source)
    labels, categories = _metric_specs(records)
    if sample_size is None:
        initial_labels = labels[:1]
    else:
        initial_labels = random.Random(seed).sample(
            labels, min(sample_size, len(labels))
        )
    page_data = {
        "title": title,
        "labels": labels,
        "selected_labels": initial_labels,
        "categories": categories,
    }
    data_json = _json_for_script(page_data)
    safe_title = html.escape(title, quote=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    html_document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; color: #252b35; background: #f6f7f9; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 30px 32px 24px; }}
    h1 {{ margin: 0 0 20px; color: #252b35; font-size: 22px; line-height: 1.25; font-weight: 650; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 11px; color: #424b59; font-size: 13px; line-height: 1.2; font-weight: 650; letter-spacing: .04em; text-transform: uppercase; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px 12px; align-items: flex-end; padding: 0 0 16px; border-bottom: 1px solid #e2e6eb; }}
    .field {{ display: grid; gap: 5px; }}
    label {{ color: #697382; font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }}
    select, button {{ font: inherit; font-size: 13px; line-height: 1.35; padding: 7px 9px; border: 1px solid #d5dbe3; border-radius: 4px; background: #ffffff; color: #252b35; }}
    select {{ min-width: 230px; }}
    select:focus, button:focus {{ outline: 2px solid #9ccbc4; outline-offset: 1px; }}
    button {{ cursor: pointer; }}
    button:hover {{ border-color: #4f968e; color: #246c65; background: #f2faf8; }}
    #label-state {{ align-self: center; min-height: 18px; color: #697382; font-size: 12px; }}
    #legend {{ display: flex; flex-wrap: wrap; gap: 7px 18px; margin: 14px 0 0; color: #596372; font-size: 12px; }}
    .legend-item {{ display: inline-flex; gap: 7px; align-items: center; }}
    .swatch {{ width: 16px; height: 3px; border-radius: 2px; }}
    .dashboard-section {{ margin-top: 25px; }}
    .section-title {{ display: flex; align-items: center; gap: 11px; margin: 0 0 11px; }}
    .section-title::after {{ content: ""; flex: 1; height: 1px; background: #e2e6eb; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr)); gap: 14px; }}
    .metric-panel {{ min-width: 0; overflow: hidden; border: 1px solid #e1e5ea; border-radius: 5px; background: #ffffff; }}
    .metric-title {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 5px 10px; padding: 12px 14px 0; color: #303846; font-size: 14px; font-weight: 650; }}
    .metric-stats {{ color: #77818f; font-size: 11px; font-weight: 500; }}
    .metric-plot {{ min-width: 0; min-height: 270px; }}
    .empty {{ color: #697382; padding: 30px 0; font-size: 13px; }}
    footer {{ color: #8a929e; font-size: 11px; padding-top: 20px; }}
    @media (max-width: 800px) {{ main {{ padding: 22px 16px 18px; }} .chart-grid {{ grid-template-columns: 1fr; }} select {{ width: 100%; }} }}
  </style>
</head>
<body>
<main>
  <h1>{safe_title}</h1>
  <div class="controls">
    <div class="field">
      <label for="label-select">Labels</label>
      <select id="label-select" multiple></select>
    </div>
    <button id="randomize" type="button">Randomize</button>
    <button id="select-all" type="button">Select all</button>
    <button id="export" type="button">Export selected HTML</button>
    <span id="label-state" aria-live="polite"></span>
  </div>
  <div id="legend" aria-label="Selected label legend"></div>
  <div id="dashboard"></div>
  <footer>AI Infra Bench &copy; 2026</footer>
</main>
<script id="metrics-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("metrics-data").textContent);
const SERIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#e377c2", "#7f7f7f"];
const SECTION_ORDER = ["Overview", "Latency", "Token Usage", "Cache", "Spec", "Finish Reason", "Other"];
const STAT_SUFFIX = new RegExp(" / (Mean|P50|P95|P99)$");
const STAT_DASH = {{ Mean: "solid", P50: "dot", P95: "dash", P99: "dashdot" }};
const labelSelect = document.getElementById("label-select");
const dashboard = document.getElementById("dashboard");
const legend = document.getElementById("legend");
const labelState = document.getElementById("label-state");

function labelColor(label) {{
  let hash = 0;
  for (let index = 0; index < label.length; index += 1) {{
    hash = ((hash << 5) - hash + label.charCodeAt(index)) | 0;
  }}
  return SERIES_COLORS[Math.abs(hash) % SERIES_COLORS.length];
}}

function escapeHtml(value) {{
  return String(value).replace(/[&<>\"']/g, (character) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }})[character]);
}}

function selectedLabels() {{
  return DATA.selected_labels.filter((label) => DATA.labels.includes(label));
}}

function syncLabelSelect() {{
  const selected = new Set(selectedLabels());
  [...labelSelect.options].forEach((option) => {{ option.selected = selected.has(option.value); }});
}}

function renderLegend(labels) {{
  legend.replaceChildren(...labels.map((label) => {{
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = labelColor(label);
    item.append(swatch, document.createTextNode(label));
    return item;
  }}));
}}

function renderMetricCategory(category, labels, categoryIndex) {{
  const visibleMetrics = category.metrics.filter((metricGroup) =>
    labels.some((label) => metricGroup.variants.some((variant) =>
      (variant.metric.series[label] || []).length
    ))
  );
  if (!visibleMetrics.length) return;
  const section = document.createElement("section");
  section.className = "dashboard-section";
  const heading = document.createElement("h2");
  heading.className = "section-title";
  heading.textContent = category.name;
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  section.append(heading, grid);
  dashboard.append(section);

  visibleMetrics.forEach((metricGroup, metricIndex) => {{
    const panel = document.createElement("div");
    panel.className = "metric-panel";
    const title = document.createElement("div");
    title.className = "metric-title";
    const titleText = document.createElement("span");
    titleText.textContent = metricGroup.name;
    title.append(titleText);
    const stats = metricGroup.variants.map((variant) => variant.stat).filter(Boolean);
    if (stats.length > 1) {{
      const statsText = document.createElement("span");
      statsText.className = "metric-stats";
      statsText.textContent = stats.join(" / ");
      title.append(statsText);
    }}
    const plot = document.createElement("div");
    plot.className = "metric-plot";
    plot.id = `plot-${{categoryIndex}}-${{metricIndex}}`;
    panel.append(title, plot);
    grid.append(panel);
    const traces = labels.flatMap((label) => metricGroup.variants.flatMap((variant) => {{
      const points = variant.metric.series[label] || [];
      if (!points.length) return [];
      const traceName = variant.stat ? `${{label}} / ${{variant.stat}}` : label;
      return [{{
        type: "scatter",
        mode: "lines+markers",
        name: traceName,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        line: {{ color: labelColor(label), width: 2.2, dash: STAT_DASH[variant.stat] || "solid" }},
        marker: {{ color: labelColor(label), size: 6 }},
        hovertemplate: "<b>" + escapeHtml(traceName) + "</b><br>concurrency=%{{x}}<br>"
          + escapeHtml(metricGroup.name) + "=%{{y}}<extra></extra>"
      }}];
    }}));
    Plotly.newPlot(plot, traces, {{
      margin: {{ l: 56, r: 16, t: 8, b: 43 }},
      height: 282,
      xaxis: {{ title: {{ text: "Concurrency", font: {{ color: "#697382", size: 10 }} }}, gridcolor: "#edf0f3", zeroline: false, linecolor: "#cfd5dc", tickfont: {{ color: "#697382", size: 10 }}, automargin: true }},
      yaxis: {{ title: {{ text: metricGroup.unit || "Value", font: {{ color: "#697382", size: 10 }} }}, gridcolor: "#edf0f3", zeroline: false, linecolor: "#cfd5dc", tickfont: {{ color: "#697382", size: 10 }}, automargin: true }},
      hovermode: "x unified",
      showlegend: false,
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      font: {{ color: "#252b35", size: 11 }},
      hoverlabel: {{ bgcolor: "#252b35", bordercolor: "#252b35", font: {{ color: "#ffffff", size: 11 }} }}
    }}, {{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] }});
  }});
}}

function groupMetrics(metrics) {{
  const groups = [];
  const percentileGroups = new Map();
  metrics.forEach((metric) => {{
    const match = metric.name.match(STAT_SUFFIX);
    const baseName = match ? metric.name.slice(0, -match[0].length) : metric.name;
    let group = match ? percentileGroups.get(baseName) : null;
    if (!group) {{
      group = {{ name: baseName, unit: metric.unit, variants: [] }};
      groups.push(group);
      if (match) percentileGroups.set(baseName, group);
    }}
    group.variants.push({{ stat: match ? match[1] : "", metric }});
  }});
  return groups;
}}

function dashboardCategories() {{
  const sections = new Map();
  DATA.categories.forEach((category) => {{
    category.metrics.forEach((metric) => {{
      const sectionNames = metric.sections || [metric.section || category.name];
      sectionNames.forEach((sectionName) => {{
        if (!sections.has(sectionName)) {{
          sections.set(sectionName, {{ name: sectionName, metrics: [] }});
        }}
        sections.get(sectionName).metrics.push(metric);
      }});
    }});
  }});
  return [...sections.values()]
    .sort((left, right) => {{
      const leftIndex = SECTION_ORDER.indexOf(left.name);
      const rightIndex = SECTION_ORDER.indexOf(right.name);
      return (leftIndex < 0 ? SECTION_ORDER.length : leftIndex)
        - (rightIndex < 0 ? SECTION_ORDER.length : rightIndex);
    }})
    .map((section) => ({{
      ...section,
      metrics: groupMetrics(section.metrics)
    }}));
}}

function render() {{
  const labels = selectedLabels();
  labelState.textContent = labels.length ? labels.join(", ") : "No labels selected";
  renderLegend(labels);
  dashboard.querySelectorAll(".metric-plot").forEach((plot) => Plotly.purge(plot));
  dashboard.replaceChildren();

  dashboardCategories().forEach((category, categoryIndex) => {{
    renderMetricCategory(category, labels, categoryIndex);
  }});

  if (!dashboard.children.length) {{
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No numeric metrics found";
    dashboard.append(empty);
  }}
}}

function randomizeLabels() {{
  const pool = [...DATA.labels];
  for (let index = pool.length - 1; index > 0; index -= 1) {{
    const swap = Math.floor(Math.random() * (index + 1));
    [pool[index], pool[swap]] = [pool[swap], pool[index]];
  }}
  DATA.selected_labels = pool.slice(0, 1);
  syncLabelSelect();
  render();
}}

function exportSelectedHtml() {{
  const selected = selectedLabels();
  const exportedData = {{
    ...DATA,
    labels: selected,
    selected_labels: selected,
    categories: DATA.categories.map((category) => ({{
      ...category,
      metrics: category.metrics.map((metric) => ({{
        ...metric,
        series: Object.fromEntries(selected.map((label) => [label, metric.series[label] || []]))
      }}))
    }}))
  }};
  const serialized = JSON.stringify(exportedData).replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026");
  const clone = document.documentElement.cloneNode(true);
  clone.querySelector("#dashboard").replaceChildren();
  clone.querySelector("#legend").replaceChildren();
  clone.querySelector("#label-select").replaceChildren();
  const source = clone.outerHTML;
  const marker = /(<script id="metrics-data" type="application\\/json">)[\\s\\S]*?(<\\/script>)/;
  const exported = source.replace(marker, (_match, prefix, suffix) => prefix + serialized + suffix);
  const blob = new Blob(["<!doctype html>\\n" + exported], {{ type: "text/html;charset=utf-8" }});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "ai-infra-bench-selected.html";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}}

DATA.labels.forEach((label) => {{
  const option = document.createElement("option");
  option.value = label;
  option.textContent = label;
  labelSelect.append(option);
}});
labelSelect.size = Math.min(6, Math.max(2, DATA.labels.length));
labelSelect.addEventListener("change", () => {{
  DATA.selected_labels = [...labelSelect.selectedOptions].map((option) => option.value);
  render();
}});
document.getElementById("randomize").addEventListener("click", randomizeLabels);
document.getElementById("select-all").addEventListener("click", () => {{
  DATA.selected_labels = [...DATA.labels];
  syncLabelSelect();
  render();
}});
document.getElementById("export").addEventListener("click", exportSelectedHtml);
syncLabelSelect();
render();
</script>
</body>
</html>
"""
    output.write_text(html_document, encoding="utf-8")
    return output


plot_metric_tables = export_metric_tables_html
export_metrics_html = export_metric_tables_html


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("must be >= 1")
    return parsed


def parse_plot_metrics_args(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser(
        prog="aib plot-metrics",
        description="Export benchmark metric tables as an interactive Plotly HTML",
    )
    parser.add_argument(
        "metrics_path",
        type=Path,
        help="Metrics JSON or JSONL produced by aib bench --metrics-path",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Output HTML path; defaults to METRICS_PATH with an .html suffix",
    )
    parser.add_argument(
        "--sample-size",
        type=_positive_int,
        default=None,
        help=SUPPRESS,
    )
    parser.add_argument("--seed", type=int, help=SUPPRESS)
    parser.add_argument(
        "--title",
        default="AI Infra Bench Metrics",
        help="HTML page and chart title",
    )
    args = parser.parse_args(argv)

    if args.metrics_path.suffix.lower() not in {".json", ".jsonl"}:
        parser.error("METRICS_PATH must end with .json or .jsonl")
    if args.output_path is None:
        args.output_path = args.metrics_path.with_suffix(".html")
    elif args.output_path.suffix.lower() != ".html":
        parser.error("--output-path must end with .html")
    return parser, args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser, args = parse_plot_metrics_args(argv)
    try:
        output_path = export_metric_tables_html(
            args.metrics_path,
            args.output_path,
            sample_size=args.sample_size,
            seed=args.seed,
            title=args.title,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported metrics HTML to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
