import html
import json
import math
import random
import re
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ai_infra_bench.utils.table import (
    fmt,
    format_histogram_percentages,
    format_mean,
    format_percentile,
    print_table,
)

__all__ = [
    "fmt",
    "format_histogram_percentages",
    "format_mean",
    "format_percentile",
    "print_table",
    "export_metric_tables_html",
    "plot_metric_tables",
    "export_metrics_html",
    "parse_plot_metrics_args",
    "main",
]

SUPPORTED_METRIC_SUFFIXES = {".yaml", ".yml", ".json", ".jsonl"}
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_LEGACY_LABEL_PATTERN = re.compile(
    r"^(?P<label>.*)\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\)$"
)
_PAREN_PERCENT_PATTERN = re.compile(r"\(([-+]?\d+(?:\.\d+)?)%\)")

_YAML_SUMMARY_LABELS = {
    "duration_s": ("Duration", "s"),
    "request_throughput_rps": ("Request Throughput", "req/s"),
    "request_goodput_rps": ("Request goodput", "req/s"),
    "input_throughput_tps": ("Input throughput", "tokens/s"),
    "output_throughput_tps": ("Output throughput", "tokens/s"),
    "total_token_throughput_tps": ("Total token throughput", "tokens/s"),
    "total_prompt_tokens": ("Total prompt tokens", "tokens"),
    "total_completion_tokens": ("Total completion tokens", "tokens"),
    "total_reasoning_tokens": ("Total reasoning tokens", "tokens"),
    "total_cached_tokens": ("Total cached tokens", "tokens"),
    "cached_tokens_device": ("Total cached tokens device", "tokens"),
    "cached_tokens_host": ("Total cached tokens host", "tokens"),
    "global_cache_ratio_pct": ("Global cache ratio", "%"),
}

_YAML_METRIC_LABELS = {
    "ttft_ms": ("TTFT", "ms"),
    "tpot_ms": ("TPOT(ecl the ttft)", "ms"),
    "itl_ms": ("ITL", "ms"),
    "latency_ms": ("Latency", "ms"),
    "prompt_tokens": ("Prompt tokens", "tokens"),
    "reasoning_tokens": ("Reasoning tokens", "tokens"),
    "cached_tokens": ("Cached tokens", "tokens"),
    "completion_tokens": ("Completion tokens", "tokens"),
    "cached_token_ratio_pct": ("Cached token ratio", "%"),
}


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
    if text.startswith("unlimited"):
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


def _convert_structured_yaml_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a human-readable YAML metric document into canonical table rows."""
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return dict(record)

    converted: Dict[str, Any] = {
        "label": record.get("label", "default"),
        "timestamp": record.get("timestamp", ""),
        "mode": record.get("mode", "concurrency"),
        "max_concurrency": record.get("max_concurrency"),
        "peak_concurrency": record.get("peak_concurrency"),
        "request_rate": record.get("request_rate"),
    }

    summary_rows: List[Dict[str, Any]] = []
    for key, (display_name, unit) in _YAML_SUMMARY_LABELS.items():
        if key not in summary or summary[key] is None:
            continue
        val = summary[key]
        if key == "request_goodput_rps" and "request_goodput_pct" in summary:
            summary_rows.append(
                {
                    "Metric": display_name,
                    "Value": f"{val} req/s ({summary['request_goodput_pct']}%)",
                }
            )
        else:
            suffix = (
                f" {unit}" if unit and unit != "%" else ("%" if unit == "%" else "")
            )
            summary_rows.append({"Metric": display_name, "Value": f"{val}{suffix}"})
    converted["Benchmark Summary"] = summary_rows

    metrics_map = record.get("metrics")
    if isinstance(metrics_map, Mapping):
        latency_rows: List[Dict[str, Any]] = []
        for key, stats in metrics_map.items():
            if not isinstance(stats, Mapping):
                continue
            display_name, unit = _YAML_METRIC_LABELS.get(key, (str(key), ""))
            row: Dict[str, Any] = {"Metric": display_name}
            for stat_key, col_name in (
                ("mean", "Mean"),
                ("p50", "P50"),
                ("p95", "P95"),
                ("p99", "P99"),
            ):
                if stats.get(stat_key) is not None:
                    row[col_name] = stats[stat_key]
            row["Unit"] = unit
            latency_rows.append(row)
        converted["Latency & Token Metrics"] = latency_rows

    spec_map = record.get("spec_tokens")
    if isinstance(spec_map, Mapping):
        spec_rows: List[Dict[str, Any]] = []
        spec_labels = {
            "total_proposed_drafts": "Total Proposed Drafts",
            "total_correct_drafts": "Total Correct Drafts",
            "total_verify_count": "Total Verify Count",
            "accept_rate_pct": "Avg Spec Accept Rate(All Verify)",
            "accept_length": "Avg Spec Accept Length(All Verify)",
            "correct_drafts_histogram": "Agg Spec Correct Drafts Histogram",
        }
        for key, value in spec_map.items():
            if value is None:
                continue
            label_text = spec_labels.get(key, str(key))
            formatted = f"{value}%" if key == "accept_rate_pct" else value
            spec_rows.append({"Metric": label_text, "Value": formatted})
        if spec_rows:
            converted["Spec Tokens Statistics"] = spec_rows

    finish_map = record.get("finish_reasons")
    if isinstance(finish_map, Mapping):
        finish_rows: List[Dict[str, Any]] = []
        for reason, details in finish_map.items():
            if isinstance(details, Mapping):
                finish_rows.append(
                    {
                        "Finish reason": str(reason),
                        "Requests": details.get("requests", 0),
                        "Percentage": f"{details.get('pct', 0)}%",
                    }
                )
        if finish_rows:
            converted["Finish Reason Statistics"] = finish_rows

    return converted


def _validate_metric_records(
    records: Any, source_name: str = "metrics source"
) -> List[Mapping[str, Any]]:
    if isinstance(records, Mapping):
        return [_convert_structured_yaml_record(records)]
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        validated: List[Mapping[str, Any]] = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"{source_name}: record {index} must be a JSON/YAML object"
                )
            validated.append(_convert_structured_yaml_record(record))
        return validated
    raise TypeError("metrics source must be a path, mapping, or sequence of mappings")


def _load_single_path(source_path: Path) -> List[Mapping[str, Any]]:
    suffix = source_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        records: List[Mapping[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as stream:
            try:
                documents = list(yaml.safe_load_all(stream))
            except yaml.YAMLError as error:
                raise ValueError(f"{source_path}: invalid YAML ({error})") from error
        for doc_index, doc in enumerate(documents, start=1):
            if doc is None:
                continue
            if isinstance(doc, list):
                records.extend(
                    _validate_metric_records(doc, f"{source_path}#{doc_index}")
                )
            elif isinstance(doc, Mapping):
                records.append(_convert_structured_yaml_record(doc))
            else:
                raise ValueError(
                    f"{source_path}: YAML document {doc_index} must be a mapping or list"
                )
        return records

    if suffix == ".jsonl":
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
                records.append(_convert_structured_yaml_record(record))
        return records

    with source_path.open("r", encoding="utf-8") as stream:
        try:
            loaded = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_path}: invalid JSON ({error.msg})") from error
    return _validate_metric_records(loaded, str(source_path))


def _load_metric_records(source: Any) -> List[Mapping[str, Any]]:
    if isinstance(source, (str, Path)):
        return _load_single_path(Path(source))
    if (
        isinstance(source, Sequence)
        and not isinstance(source, (str, bytes))
        and source
        and all(isinstance(item, (str, Path)) for item in source)
    ):
        merged: List[Mapping[str, Any]] = []
        for item in source:
            merged.extend(_load_single_path(Path(item)))
        return merged
    return _validate_metric_records(source)


def _normalize_label(raw_label: Any, fallback_index: int) -> str:
    label = str(raw_label or f"run-{fallback_index}")
    if legacy_match := _LEGACY_LABEL_PATTERN.fullmatch(label):
        return legacy_match.group("label") or "benchmark"
    return label


def _metric_sections(table_title: str, metric_name: str) -> List[str]:
    table = table_title.lower()
    metric = metric_name.lower()
    sections: List[str] = []
    is_summary = "benchmark summary" in table
    if is_summary and metric in {
        "duration",
        "output throughput",
        "input throughput",
        "total token throughput",
        "request throughput",
        "request goodput",
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
    elif any(token in metric for token in ("ttft", "tpot", "itl", "latency")):
        sections.append("Latency")
    if is_summary and not sections:
        return []
    return list(dict.fromkeys(sections or ["Other"]))


def _normalize_table_title(table_title: str) -> str:
    if table_title.startswith("Benchmark Summary ("):
        return "Benchmark Summary"
    return table_title


def _extract_load_axes(
    record: Mapping[str, Any], fallback_index: int
) -> tuple[float | None, float | None]:
    concurrency_val = _parse_plot_value(record.get("max_concurrency"))
    if not isinstance(concurrency_val, float):
        peak_val = _parse_plot_value(record.get("peak_concurrency"))
        concurrency_val = peak_val if isinstance(peak_val, float) else None

    qps_val = _parse_plot_value(record.get("request_rate"))
    if not isinstance(qps_val, float):
        qps_val = None

    summary_rows = next(
        (
            rows
            for title, rows in record.items()
            if _normalize_table_title(str(title)) == "Benchmark Summary"
            and isinstance(rows, list)
        ),
        [],
    )
    for row in summary_rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("Metric", "")).lower()
        if key == "max concurrency" and concurrency_val is None:
            parsed = _parse_plot_value(row.get("Value"))
            if isinstance(parsed, float):
                concurrency_val = parsed
        elif key == "request rate" and qps_val is None:
            parsed = _parse_plot_value(row.get("Value"))
            if isinstance(parsed, float):
                qps_val = parsed

    if concurrency_val is None and qps_val is None:
        concurrency_val = float(fallback_index)
    return concurrency_val, qps_val


def _metric_specs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[List[str], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels: List[str] = []
    categories: Dict[str, Dict[str, Any]] = {}
    run_rows: List[Dict[str, Any]] = []

    qps_modes = 0
    distinct_qps: set[float] = set()
    distinct_conc: set[float] = set()
    resolved_axes: List[tuple[str, float | None, float | None]] = []

    for idx, record in enumerate(records, start=1):
        label = _normalize_label(record.get("label") or record.get("Label"), idx)
        if label not in labels:
            labels.append(label)
        conc_val, qps_val = _extract_load_axes(record, idx)
        if record.get("mode") == "qps" or (
            qps_val is not None
            and str(record.get("max_concurrency", "")).startswith("unlimited")
        ):
            qps_modes += 1
        if qps_val is not None:
            distinct_qps.add(qps_val)
        if conc_val is not None:
            distinct_conc.add(conc_val)
        resolved_axes.append((label, conc_val, qps_val))

    prefer_qps = qps_modes > 0 or (len(distinct_qps) > 1 and len(distinct_conc) <= 1)
    default_x_mode = "qps" if prefer_qps else "concurrency"

    def add_point(
        category: str,
        metric_name: str,
        unit: str,
        label: str,
        primary_x: float,
        conc_val: float | None,
        qps_val: float | None,
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
        bucket = label_samples.setdefault(
            primary_x, {"ys": [], "conc": conc_val, "qps": qps_val}
        )
        bucket["ys"].append(value)

    for record_index, (record, (label, conc_val, qps_val)) in enumerate(
        zip(records, resolved_axes), start=1
    ):
        primary_x = (
            (qps_val if qps_val is not None else (conc_val or float(record_index)))
            if prefer_qps
            else (
                conc_val if conc_val is not None else (qps_val or float(record_index))
            )
        )
        flat_summary: Dict[str, Any] = {
            "label": label,
            "timestamp": str(record.get("timestamp") or ""),
            "concurrency": conc_val,
            "qps": qps_val,
            "x": primary_x,
        }

        for table_title, rows in record.items():
            if not isinstance(rows, list):
                continue
            category = _normalize_table_title(str(table_title))
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
                    sections = _metric_sections(category, str(row_label))
                    if not sections:
                        continue
                    if isinstance(value, list):
                        for bin_index, bin_value in enumerate(value):
                            add_point(
                                category,
                                f"{metric_name} / bin {bin_index}",
                                unit,
                                label,
                                primary_x,
                                conc_val,
                                qps_val,
                                bin_value,
                                sections,
                            )
                    else:
                        flat_summary[metric_name] = value
                        add_point(
                            category,
                            metric_name,
                            unit,
                            label,
                            primary_x,
                            conc_val,
                            qps_val,
                            value,
                            sections,
                        )
                        if ratio_match := _PAREN_PERCENT_PATTERN.search(str(raw_value)):
                            ratio_val = float(ratio_match.group(1))
                            flat_summary[f"{metric_name} / Ratio"] = ratio_val
                            add_point(
                                category,
                                f"{metric_name} / Ratio",
                                "%",
                                label,
                                primary_x,
                                conc_val,
                                qps_val,
                                ratio_val,
                                sections,
                            )
        run_rows.append(flat_summary)

    category_order = [
        "Benchmark Summary",
        "Latency & Token Metrics",
        "Spec Tokens Statistics",
        "Finish Reason Statistics",
    ]
    category_order.extend(c for c in categories if c not in category_order)

    normalized_categories = []
    for category_name in category_order:
        category = categories.get(category_name)
        if category is None:
            continue
        metrics = []
        for metric in category["metrics"].values():
            series = {}
            for label, samples in metric.pop("samples").items():
                points = []
                for x_val, bucket in sorted(samples.items()):
                    point: Dict[str, Any] = {
                        "x": x_val,
                        "y": sum(bucket["ys"]) / len(bucket["ys"]),
                    }
                    if bucket["conc"] is not None:
                        point["concurrency"] = bucket["conc"]
                    if bucket["qps"] is not None:
                        point["qps"] = bucket["qps"]
                    points.append(point)
                series[label] = points
            metrics.append({**metric, "series": series})
        normalized_categories.append({"name": category["name"], "metrics": metrics})

    return labels, default_x_mode, normalized_categories, run_rows


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
    labels_filter: Sequence[str] | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    title: str = "AI Infra Bench Report",
) -> Path:
    """Export YAML/JSON/JSONL benchmark metrics as an interactive HTML dashboard."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    records = _load_metric_records(metrics_source)

    if labels_filter:
        wanted = {item.strip() for item in labels_filter if item.strip()}
        available_all = list(
            dict.fromkeys(
                _normalize_label(r.get("label") or r.get("Label"), idx)
                for idx, r in enumerate(records, start=1)
            )
        )
        records = [
            r
            for idx, r in enumerate(records, start=1)
            if _normalize_label(r.get("label") or r.get("Label"), idx) in wanted
        ]
        if not records:
            raise ValueError(
                f"no records matched --labels {sorted(wanted)}; "
                f"available labels: {available_all}"
            )

    labels, default_x_mode, categories, run_rows = _metric_specs(records)
    if sample_size is None:
        initial_labels = labels[:8]
    else:
        initial_labels = random.Random(seed).sample(
            labels, min(sample_size, len(labels))
        )

    page_data = {
        "title": title,
        "default_x_mode": default_x_mode,
        "labels": labels,
        "selected_labels": initial_labels,
        "categories": categories,
        "runs": run_rows,
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
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-soft: #f1f5f9;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --text: #0f172a;
      --text-muted: #64748b;
      --brand: #0284c7;
      --brand-soft: rgba(2, 132, 199, 0.1);
      --grid: #f1f5f9;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px -4px rgba(15, 23, 42, 0.04);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    html.dark {{
      --bg: #0b1120;
      --surface: #111827;
      --surface-soft: #1e293b;
      --border: #1f2937;
      --border-strong: #334155;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --brand: #38bdf8;
      --brand-soft: rgba(56, 189, 248, 0.14);
      --grid: #1e293b;
      --shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--text); background: var(--bg); transition: background .15s ease, color .15s ease; }}
    header {{
      position: sticky; top: 0; z-index: 20;
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
    }}
    .header-inner {{
      max-width: 1720px; margin: 0 auto; padding: 14px 28px;
      display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; justify-content: space-between;
    }}
    .brand {{ display: flex; align-items: center; gap: 10px; }}
    .brand-mark {{
      width: 28px; height: 28px; border-radius: 8px;
      background: linear-gradient(135deg, #0284c7, #0ea5e9 55%, #14b8a6);
      display: inline-flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 13px;
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; letter-spacing: -0.02em; }}
    .toolbar-actions {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }}
    .seg-group {{
      display: inline-flex; background: var(--surface-soft); padding: 3px; border-radius: 8px; border: 1px solid var(--border);
    }}
    .seg-btn {{
      border: none; background: transparent; color: var(--text-muted); font: inherit; font-size: 12px; font-weight: 550;
      padding: 4px 10px; border-radius: 6px; cursor: pointer; transition: all .12s ease;
    }}
    .seg-btn.active {{ background: var(--surface); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
    .btn {{
      font: inherit; font-size: 12px; font-weight: 550; padding: 6px 11px; border-radius: 8px;
      border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer;
    }}
    .btn:hover {{ border-color: var(--brand); color: var(--brand); }}
    main {{ max-width: 1720px; margin: 0 auto; padding: 20px 28px 36px; }}
    .filter-bar {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 16px; box-shadow: var(--shadow); display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center;
    }}
    .filter-label {{ font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: .06em; color: var(--text-muted); }}
    #label-chips {{ display: flex; flex-wrap: wrap; gap: 7px; flex: 1; min-width: 220px; }}
    .chip {{
      display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px; border-radius: 999px;
      font-size: 12.5px; font-weight: 550; cursor: pointer; user-select: none;
      border: 1px solid var(--border); background: var(--surface-soft); color: var(--text-muted); transition: all .12s ease;
    }}
    .chip.active {{ background: var(--surface); border-color: var(--border-strong); color: var(--text); box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .chip-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; opacity: 0.35; }}
    .chip.active .chip-dot {{ opacity: 1; }}
    #chip-search {{
      font: inherit; font-size: 12px; padding: 5px 10px; border-radius: 7px;
      border: 1px solid var(--border); background: var(--bg); color: var(--text); width: 150px;
    }}
    .kpi-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 16px;
    }}
    .kpi-card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; box-shadow: var(--shadow);
    }}
    .kpi-label {{ font-size: 11.5px; color: var(--text-muted); font-weight: 550; }}
    .kpi-value {{ font-size: 22px; font-weight: 700; letter-spacing: -0.02em; margin: 5px 0 4px; font-variant-numeric: tabular-nums; }}
    .kpi-sub {{ font-size: 11.5px; color: var(--brand); font-weight: 550; }}
    .dashboard-section {{ margin-top: 26px; }}
    .section-title {{
      display: flex; align-items: center; gap: 12px; margin: 0 0 12px;
      font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted);
    }}
    .section-title::after {{ content: ""; flex: 1; height: 1px; background: var(--border); }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 480px), 1fr)); gap: 14px; }}
    .metric-panel {{
      min-width: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 12px;
      background: var(--surface); box-shadow: var(--shadow); display: flex; flex-direction: column;
    }}
    .metric-header {{
      display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 13px 16px 4px;
    }}
    .metric-title {{ font-size: 14px; font-weight: 650; color: var(--text); letter-spacing: -0.01em; }}
    .unit-badge {{
      font-size: 11px; font-weight: 600; color: var(--text-muted); background: var(--surface-soft);
      padding: 2px 7px; border-radius: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .metric-plot {{ min-width: 0; min-height: 295px; }}
    .table-wrap {{
      overflow-x: auto; background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; box-shadow: var(--shadow);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }}
    th, td {{ padding: 9px 13px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th {{ font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); background: var(--surface-soft); }}
    tr:last-child td {{ border-bottom: none; }}
    .empty {{ color: var(--text-muted); padding: 36px 0; text-align: center; font-size: 14px; }}
    footer {{ color: var(--text-muted); font-size: 12px; padding-top: 28px; text-align: center; }}
    @media (max-width: 820px) {{
      .header-inner, main {{ padding-left: 14px; padding-right: 14px; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <span class="brand-mark">&gt;_</span>
      <h1>{safe_title}</h1>
    </div>
    <div class="toolbar-actions">
      <div class="seg-group" id="xaxis-tabs" title="X-Axis Variable">
        <button type="button" class="seg-btn active" data-xmode="auto">Auto X</button>
        <button type="button" class="seg-btn" data-xmode="concurrency">Concurrency</button>
        <button type="button" class="seg-btn" data-xmode="qps">QPS</button>
      </div>
      <div class="seg-group" id="stat-tabs" title="Latency & Token Percentiles">
        <button type="button" class="seg-btn active" data-stat="Mean,P99">Mean + P99</button>
        <button type="button" class="seg-btn" data-stat="Mean">Mean</button>
        <button type="button" class="seg-btn" data-stat="P50">P50</button>
        <button type="button" class="seg-btn" data-stat="P95">P95</button>
        <button type="button" class="seg-btn" data-stat="P99">P99</button>
        <button type="button" class="seg-btn" data-stat="ALL">All</button>
      </div>
      <button id="export" class="btn" type="button">Export selected HTML</button>
      <button id="theme-toggle" class="btn" type="button" title="Toggle Dark/Light Theme">Theme</button>
    </div>
  </div>
</header>
<main>
  <div class="filter-bar">
    <span class="filter-label">Labels</span>
    <input id="chip-search" type="search" placeholder="Filter labels..." hidden>
    <div id="label-chips"></div>
    <select id="label-select" multiple hidden></select>
    <div style="display:flex;gap:6px;">
      <button id="select-all" class="btn" type="button">Select all</button>
      <button id="clear-all" class="btn" type="button">Clear</button>
      <button id="randomize" class="btn" type="button">Randomize</button>
    </div>
  </div>
  <div id="kpi-strip" class="kpi-grid"></div>
  <div id="dashboard"></div>
  <footer>AI Infra Bench &copy; 2026</footer>
</main>
<script id="metrics-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("metrics-data").textContent);
const PALETTE = ["#0284c7", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#eab308", "#06b6d4", "#ec4899", "#6366f1", "#14b8a6"];
const SECTION_ORDER = ["Overview", "Latency", "Token Usage", "Cache", "Spec", "Finish Reason", "Other"];
const STAT_SUFFIX = new RegExp(" / (Mean|P50|P95|P99)$");
const STAT_DASH = {{ Mean: "solid", P50: "dot", P95: "dash", P99: "dashdot" }};

let xAxisMode = "auto";
let activeStats = new Set(["Mean", "P99"]);
let chipQuery = "";

const labelSelect = document.getElementById("label-select");
const labelChips = document.getElementById("label-chips");
const chipSearch = document.getElementById("chip-search");
const kpiStrip = document.getElementById("kpi-strip");
const dashboard = document.getElementById("dashboard");

function labelColor(label) {{
  const idx = DATA.labels.indexOf(label);
  return PALETTE[(idx >= 0 ? idx : 0) % PALETTE.length];
}}

function escapeHtml(value) {{
  return String(value).replace(/[&<>\"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}})[ch]);
}}

function selectedLabels() {{
  return DATA.selected_labels.filter((label) => DATA.labels.includes(label));
}}

function effectiveXMode() {{
  return xAxisMode === "auto" ? (DATA.default_x_mode || "concurrency") : xAxisMode;
}}

function pointX(point) {{
  const mode = effectiveXMode();
  if (mode === "qps" && point.qps != null) return point.qps;
  if (mode === "concurrency" && point.concurrency != null) return point.concurrency;
  return point.x;
}}

function xAxisTitle() {{
  return effectiveXMode() === "qps" ? "Request Rate (QPS)" : "Concurrency";
}}

function themeColors() {{
  const dark = document.documentElement.classList.contains("dark");
  return dark
    ? {{ paper: "#111827", plot: "#111827", text: "#e2e8f0", muted: "#94a3b8", grid: "#1f2937", line: "#334155", hoverBg: "#0f172a" }}
    : {{ paper: "#ffffff", plot: "#ffffff", text: "#0f172a", muted: "#64748b", grid: "#f1f5f9", line: "#cbd5e1", hoverBg: "#0f172a" }};
}}

function basePlotlyLayout(xTitle, yTitle, showLegend = true) {{
  const c = themeColors();
  return {{
    margin: {{ l: 58, r: 18, t: 10, b: showLegend ? 54 : 42 }},
    height: 300,
    xaxis: {{ title: {{ text: xTitle, font: {{ color: c.muted, size: 11 }} }}, gridcolor: c.grid, zeroline: false, linecolor: c.line, tickfont: {{ color: c.muted, size: 11 }}, automargin: true }},
    yaxis: {{ title: {{ text: yTitle, font: {{ color: c.muted, size: 11 }} }}, gridcolor: c.grid, zeroline: false, linecolor: c.line, tickfont: {{ color: c.muted, size: 11 }}, automargin: true }},
    hovermode: "closest",
    showlegend: showLegend,
    legend: {{ orientation: "h", y: -0.22, x: 0, font: {{ color: c.muted, size: 11 }} }},
    paper_bgcolor: c.paper,
    plot_bgcolor: c.plot,
    font: {{ color: c.text, size: 11.5 }},
    hoverlabel: {{ bgcolor: c.hoverBg, bordercolor: c.hoverBg, font: {{ color: "#ffffff", size: 11.5 }} }}
  }};
}}

function syncControls() {{
  const selected = new Set(selectedLabels());
  [...labelSelect.options].forEach((opt) => {{ opt.selected = selected.has(opt.value); }});
  labelChips.replaceChildren(...DATA.labels
    .filter((l) => !chipQuery || l.toLowerCase().includes(chipQuery))
    .map((label) => {{
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip" + (selected.has(label) ? " active" : "");
      const dot = document.createElement("span");
      dot.className = "chip-dot";
      dot.style.background = labelColor(label);
      btn.append(dot, document.createTextNode(label));
      btn.addEventListener("click", () => {{
        if (selected.has(label)) {{
          DATA.selected_labels = DATA.selected_labels.filter((item) => item !== label);
        }} else {{
          DATA.selected_labels = [...DATA.selected_labels, label];
        }}
        syncControls();
        render();
      }});
      return btn;
    }}));
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
      (metric.sections || [metric.section || category.name]).forEach((sectionName) => {{
        if (!sections.has(sectionName)) sections.set(sectionName, {{ name: sectionName, metrics: [] }});
        sections.get(sectionName).metrics.push(metric);
      }});
    }});
  }});
  return [...sections.values()]
    .sort((a, b) => {{
      const ai = SECTION_ORDER.indexOf(a.name);
      const bi = SECTION_ORDER.indexOf(b.name);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
    }})
    .map((s) => ({{ ...s, metrics: groupMetrics(s.metrics) }}));
}}

function renderKpis(labels) {{
  kpiStrip.replaceChildren();
  const runs = (DATA.runs || []).filter((r) => labels.includes(r.label));
  if (!runs.length) return;

  const bestMax = (key, unit, title) => {{
    const valid = runs.filter((r) => typeof r[key] === "number");
    if (!valid.length) return null;
    const winner = valid.reduce((a, b) => (b[key] > a[key] ? b : a));
    return {{ title, value: `${{winner[key].toFixed(2)}} ${{unit}}`, sub: winner.label }};
  }};
  const bestMin = (key, unit, title) => {{
    const valid = runs.filter((r) => typeof r[key] === "number" && r[key] > 0);
    if (!valid.length) return null;
    const winner = valid.reduce((a, b) => (b[key] < a[key] ? b : a));
    return {{ title, value: `${{winner[key].toFixed(2)}} ${{unit}}`, sub: winner.label }};
  }};

  [
    bestMax("Output throughput", "tok/s", "Peak Output Throughput"),
    bestMax("Total token throughput", "tok/s", "Peak Total Throughput"),
    bestMin("TTFT / Mean", "ms", "Lowest Mean TTFT"),
    bestMin("TPOT(ecl the ttft) / Mean", "ms", "Lowest Mean TPOT")
  ].filter(Boolean).forEach((item) => {{
    const card = document.createElement("div");
    card.className = "kpi-card";
    card.innerHTML = `<div class="kpi-label">${{escapeHtml(item.title)}}</div>`
      + `<div class="kpi-value">${{escapeHtml(item.value)}}</div>`
      + `<div class="kpi-sub">${{escapeHtml(item.sub)}}</div>`;
    kpiStrip.append(card);
  }});
}}

function renderParetoSection(labels) {{
  const runs = (DATA.runs || []).filter((r) => labels.includes(r.label));
  if (runs.length < 2) return;

  const preferredStat = activeStats.has("P99") && activeStats.size === 1 ? "P99"
    : activeStats.has("P95") && activeStats.size === 1 ? "P95"
    : activeStats.has("P50") && activeStats.size === 1 ? "P50" : "Mean";

  const specs = [
    {{
      title: "Pareto Frontier: Output Throughput vs. User Speed",
      unit: "tok/s",
      xTitle: "Interactivity (1000 / Mean TPOT, tok/s/user)",
      yTitle: "Output Throughput (tokens/s)",
      extract: (r) => {{
        const tpot = r["TPOT(ecl the ttft) / Mean"];
        const outTps = r["Output throughput"];
        return (tpot > 0 && typeof outTps === "number") ? {{ x: 1000.0 / tpot, y: outTps }} : null;
      }}
    }},
    {{
      title: `Throughput vs. TTFT (${{preferredStat}})`,
      unit: "ms",
      xTitle: "Output Throughput (tokens/s)",
      yTitle: `TTFT ${{preferredStat}} (ms)`,
      extract: (r) => {{
        const ttft = r[`TTFT / ${{preferredStat}}`] ?? r["TTFT / Mean"];
        const outTps = r["Output throughput"];
        return (typeof ttft === "number" && typeof outTps === "number") ? {{ x: outTps, y: ttft }} : null;
      }}
    }}
  ];

  const validSpecs = specs.filter((spec) => runs.some((r) => spec.extract(r) !== null));
  if (!validSpecs.length) return;

  const section = document.createElement("section");
  section.className = "dashboard-section";
  const heading = document.createElement("h2");
  heading.className = "section-title";
  heading.textContent = "Trade-off & Pareto Frontier";
  const grid = document.createElement("div");
  grid.className = "chart-grid";
  section.append(heading, grid);
  dashboard.append(section);

  validSpecs.forEach((spec, idx) => {{
    const panel = document.createElement("div");
    panel.className = "metric-panel";
    const header = document.createElement("div");
    header.className = "metric-header";
    const titleEl = document.createElement("div");
    titleEl.className = "metric-title";
    titleEl.textContent = spec.title;
    const badge = document.createElement("span");
    badge.className = "unit-badge";
    badge.textContent = spec.unit;
    header.append(titleEl, badge);
    const plot = document.createElement("div");
    plot.className = "metric-plot";
    plot.id = `pareto-${{idx}}`;
    panel.append(header, plot);
    grid.append(panel);

    const traces = labels.flatMap((label) => {{
      const pts = runs
        .filter((r) => r.label === label)
        .map((r) => {{
          const xy = spec.extract(r);
          return xy ? {{ ...xy, conc: r.concurrency, qps: r.qps }} : null;
        }})
        .filter(Boolean)
        .sort((a, b) => a.x - b.x);
      if (!pts.length) return [];
      return [{{
        type: "scatter",
        mode: "lines+markers",
        name: label,
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        customdata: pts.map((p) => `conc=${{p.conc ?? "unlimited"}}, qps=${{p.qps ?? "inf"}}`),
        line: {{ color: labelColor(label), width: 2.4 }},
        marker: {{ color: labelColor(label), size: 7 }},
        hovertemplate: `<b>${{escapeHtml(label)}}</b> (%{{customdata}})<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>`
      }}];
    }});

    Plotly.newPlot(plot, traces, basePlotlyLayout(spec.xTitle, spec.yTitle, true), {{
      responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }});
  }});
}}

function renderMetricCategory(category, labels, categoryIndex) {{
  const visibleMetrics = category.metrics.filter((metricGroup) =>
    labels.some((label) => metricGroup.variants.some((variant) =>
      (!variant.stat || activeStats.has("ALL") || activeStats.has(variant.stat))
      && (variant.metric.series[label] || []).length
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
    const header = document.createElement("div");
    header.className = "metric-header";
    const title = document.createElement("div");
    title.className = "metric-title";
    title.textContent = metricGroup.name;
    header.append(title);
    if (metricGroup.unit) {{
      const badge = document.createElement("span");
      badge.className = "unit-badge";
      badge.textContent = metricGroup.unit;
      header.append(badge);
    }}
    const plot = document.createElement("div");
    plot.className = "metric-plot";
    plot.id = `plot-${{categoryIndex}}-${{metricIndex}}`;
    panel.append(header, plot);
    grid.append(panel);

    const filteredVariants = metricGroup.variants.filter(
      (v) => !v.stat || activeStats.has("ALL") || activeStats.has(v.stat)
    );
    const traces = labels.flatMap((label) => filteredVariants.flatMap((variant) => {{
      const rawPoints = variant.metric.series[label] || [];
      if (!rawPoints.length) return [];
      const sorted = [...rawPoints].sort((a, b) => pointX(a) - pointX(b));
      const traceName = variant.stat && filteredVariants.length > 1 ? `${{label}} (${{variant.stat}})` : label;
      return [{{
        type: "scatter",
        mode: "lines+markers",
        name: traceName,
        x: sorted.map((p) => pointX(p)),
        y: sorted.map((p) => p.y),
        line: {{ color: labelColor(label), width: 2.3, dash: STAT_DASH[variant.stat] || "solid" }},
        marker: {{ color: labelColor(label), size: 6.5 }},
        hovertemplate: `<b>${{escapeHtml(traceName)}}</b><br>${{escapeHtml(xAxisTitle())}}=%{{x}}<br>${{escapeHtml(metricGroup.name)}}=%{{y:.2f}} ${{escapeHtml(metricGroup.unit || "")}}<extra></extra>`
      }}];
    }}));

    const layout = basePlotlyLayout(xAxisTitle(), metricGroup.unit || "Value", traces.length <= 10);
    layout.hovermode = "x unified";
    Plotly.newPlot(plot, traces, layout, {{
      responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"]
    }});
  }});
}}

function renderSummaryTable(labels) {{
  const runs = (DATA.runs || []).filter((r) => labels.includes(r.label));
  if (!runs.length) return;
  const section = document.createElement("section");
  section.className = "dashboard-section";
  const heading = document.createElement("h2");
  heading.className = "section-title";
  heading.textContent = "Run Summary Table";
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const fmtNum = (v, d = 2) => (typeof v === "number" ? v.toFixed(d) : "—");
  const rowsHtml = runs.map((r) => `<tr>
    <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{labelColor(r.label)}};margin-right:7px;"></span><b>${{escapeHtml(r.label)}}</b></td>
    <td>${{r.concurrency ?? "unlimited"}}</td>
    <td>${{r.qps ?? "inf"}}</td>
    <td>${{fmtNum(r["Output throughput"])}}</td>
    <td>${{fmtNum(r["Total token throughput"])}}</td>
    <td>${{fmtNum(r["Request Throughput"] ?? r["Mean finished requests per second"])}}</td>
    <td>${{fmtNum(r["TTFT / Mean"])}} / ${{fmtNum(r["TTFT / P99"])}}</td>
    <td>${{fmtNum(r["TPOT(ecl the ttft) / Mean"])}} / ${{fmtNum(r["TPOT(ecl the ttft) / P99"])}}</td>
    <td>${{fmtNum(r["Latency / Mean"])}} / ${{fmtNum(r["Latency / P99"])}}</td>
  </tr>`).join("");
  wrap.innerHTML = `<table>
    <thead><tr>
      <th>Label</th><th>Concurrency</th><th>QPS</th><th>Output tok/s</th><th>Total tok/s</th>
      <th>Req/s</th><th>TTFT Mean / P99 (ms)</th><th>TPOT Mean / P99 (ms)</th><th>E2E Mean / P99 (ms)</th>
    </tr></thead>
    <tbody>${{rowsHtml}}</tbody>
  </table>`;
  section.append(heading, wrap);
  dashboard.append(section);
}}

function render() {{
  const labels = selectedLabels();
  dashboard.querySelectorAll(".metric-plot").forEach((plot) => Plotly.purge(plot));
  dashboard.replaceChildren();
  renderKpis(labels);
  renderParetoSection(labels);
  dashboardCategories().forEach((category, idx) => renderMetricCategory(category, labels, idx));
  renderSummaryTable(labels);

  if (!dashboard.children.length) {{
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = labels.length ? "No numeric metrics found" : "Select one or more labels above to compare runs";
    dashboard.append(empty);
  }}
}}

function randomizeLabels() {{
  const pool = [...DATA.labels];
  for (let i = pool.length - 1; i > 0; i -= 1) {{
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }}
  DATA.selected_labels = pool.slice(0, Math.min(2, pool.length));
  syncControls();
  render();
}}

function exportSelectedHtml() {{
  const selected = selectedLabels();
  const exportedData = {{
    ...DATA,
    labels: selected,
    selected_labels: selected,
    runs: (DATA.runs || []).filter((r) => selected.includes(r.label)),
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
  clone.querySelector("#kpi-strip").replaceChildren();
  clone.querySelector("#label-chips").replaceChildren();
  clone.querySelector("#label-select").replaceChildren();
  const source = clone.outerHTML;
  const marker = /(<script id="metrics-data" type="application\\/json">)[\\s\\S]*?(<\\/script>)/;
  const exported = source.replace(marker, (_m, prefix, suffix) => prefix + serialized + suffix);
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
  const opt = document.createElement("option");
  opt.value = label;
  opt.textContent = label;
  labelSelect.append(opt);
}});
if (DATA.labels.length > 6) {{
  chipSearch.hidden = false;
  chipSearch.addEventListener("input", () => {{
    chipQuery = chipSearch.value.trim().toLowerCase();
    syncControls();
  }});
}}
document.querySelectorAll("#xaxis-tabs .seg-btn").forEach((btn) => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#xaxis-tabs .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    xAxisMode = btn.dataset.xmode;
    render();
  }});
}});
document.querySelectorAll("#stat-tabs .seg-btn").forEach((btn) => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#stat-tabs .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeStats = new Set(btn.dataset.stat.split(","));
    render();
  }});
}});
document.getElementById("select-all").addEventListener("click", () => {{
  DATA.selected_labels = [...DATA.labels];
  syncControls();
  render();
}});
document.getElementById("clear-all").addEventListener("click", () => {{
  DATA.selected_labels = [];
  syncControls();
  render();
}});
document.getElementById("randomize").addEventListener("click", randomizeLabels);
document.getElementById("export").addEventListener("click", exportSelectedHtml);
document.getElementById("theme-toggle").addEventListener("click", () => {{
  document.documentElement.classList.toggle("dark");
  render();
}});
syncControls();
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
        description="Render YAML/JSON/JSONL benchmark metrics into an interactive HTML dashboard",
    )
    parser.add_argument(
        "metrics_paths",
        nargs="+",
        type=Path,
        metavar="METRICS_PATH",
        help="One or more .yaml/.yml/.json/.jsonl metric files produced by aib bench",
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Only include specific benchmark labels from the input YAML/JSON files",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Output HTML path (defaults to <first-input>.html or metrics-comparison.html)",
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
        default="AI Infra Bench Report",
        help="HTML page and dashboard title",
    )
    args = parser.parse_args(argv)

    for path in args.metrics_paths:
        if path.suffix.lower() not in SUPPORTED_METRIC_SUFFIXES:
            parser.error(
                f"{path}: METRICS_PATH must end with .yaml, .yml, .json, or .jsonl"
            )
    if args.output_path is None:
        args.output_path = (
            args.metrics_paths[0].with_suffix(".html")
            if len(args.metrics_paths) == 1
            else Path("metrics-comparison.html")
        )
    elif args.output_path.suffix.lower() != ".html":
        parser.error("--output-path must end with .html")
    args.metrics_path = args.metrics_paths[0]
    return parser, args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser, args = parse_plot_metrics_args(argv)
    source: Any = (
        args.metrics_paths[0] if len(args.metrics_paths) == 1 else args.metrics_paths
    )
    try:
        output_path = export_metric_tables_html(
            source,
            args.output_path,
            labels_filter=args.labels,
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
