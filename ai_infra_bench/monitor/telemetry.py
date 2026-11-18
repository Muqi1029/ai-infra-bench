"""Prometheus text parsing, service fingerprinting, and Grafana/platform helpers."""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Sequence

from ai_infra_bench.utils.req import api_url, sanitize_url

logger = logging.getLogger(__name__)

_METRIC_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)

# Canonical metric aliases across SGLang and vLLM Prometheus expositions.
_KV_CACHE_KEYS = (
    "sglang:token_usage",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
)
_RUNNING_REQ_KEYS = (
    "sglang:num_running_reqs",
    "vllm:num_requests_running",
)
_WAITING_REQ_KEYS = (
    "sglang:num_queue_reqs",
    "vllm:num_requests_waiting",
)
_GEN_THROUGHPUT_KEYS = (
    "sglang:gen_throughput",
    "vllm:avg_generation_throughput_toks_per_s",
)
_PROMPT_THROUGHPUT_KEYS = (
    "sglang:prompt_throughput",
    "vllm:avg_prompt_throughput_toks_per_s",
)
_CACHE_HIT_KEYS = (
    "sglang:cache_hit_rate",
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:prefix_cache_hit_rate",
)
_SPEC_ACCEPT_LEN_KEYS = ("sglang:spec_accept_length",)
_SPEC_ACCEPT_RATE_KEYS = ("sglang:spec_accept_rate",)
_GEN_TOKENS_COUNTER_KEYS = (
    "sglang:generation_tokens_total",
    "vllm:generation_tokens_total",
)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def opener_for_url(url: str) -> urllib.request.OpenerDirector:
    parsed = urllib.parse.urlsplit(url)
    if _is_loopback_host(parsed.hostname):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _normalize_ratio_pct(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    # Both SGLang token_usage and vLLM gpu_cache_usage_perc report [0.0, 1.0];
    # handle already-scaled [0, 100] values gracefully if > 1.0001.
    scaled = value * 100.0 if 0.0 <= value <= 1.0001 else value
    return round(max(0.0, min(100.0, scaled)), 2)


def parse_prometheus_text(metrics_text: str) -> dict[str, Any]:
    """Extract normalized LLM serving gauges and counters from Prometheus text."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue
        name, _labels, raw_val = match.groups()
        try:
            value = float(raw_val)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    def first_avg(keys: Sequence[str]) -> float | None:
        for key in keys:
            if key in sums and counts.get(key, 0) > 0:
                return sums[key] / counts[key]
        return None

    def first_sum(keys: Sequence[str]) -> float | None:
        for key in keys:
            if key in sums:
                return sums[key]
        return None

    kv_pct = _normalize_ratio_pct(first_avg(_KV_CACHE_KEYS))
    cache_hit_pct = _normalize_ratio_pct(first_avg(_CACHE_HIT_KEYS))
    running = first_sum(_RUNNING_REQ_KEYS)
    waiting = first_sum(_WAITING_REQ_KEYS)
    gen_tps = first_sum(_GEN_THROUGHPUT_KEYS)
    prompt_tps = first_sum(_PROMPT_THROUGHPUT_KEYS)
    spec_len = first_avg(_SPEC_ACCEPT_LEN_KEYS)
    spec_rate = _normalize_ratio_pct(first_avg(_SPEC_ACCEPT_RATE_KEYS))
    gen_tokens_total = first_sum(_GEN_TOKENS_COUNTER_KEYS)

    return {
        "kv_cache_pct": kv_pct,
        "running_reqs": int(round(running)) if running is not None else None,
        "waiting_reqs": int(round(waiting)) if waiting is not None else None,
        "gen_throughput_tps": round(gen_tps, 2) if gen_tps is not None else None,
        "prompt_throughput_tps": (
            round(prompt_tps, 2) if prompt_tps is not None else None
        ),
        "cache_hit_pct": cache_hit_pct,
        "spec_accept_length": round(spec_len, 2) if spec_len is not None else None,
        "spec_accept_rate_pct": spec_rate,
        "gen_tokens_total": gen_tokens_total,
    }


def scrape_metrics_point(
    metrics_url: str, timeout: float = 2.0
) -> dict[str, Any] | None:
    """Fetch and parse a single Prometheus `/metrics` endpoint."""
    req = urllib.request.Request(
        metrics_url, headers={"User-Agent": "ai-infra-bench-monitoring"}
    )
    try:
        with opener_for_url(metrics_url).open(req, timeout=timeout) as resp:
            if not 200 <= resp.status < 300:
                return None
            text = resp.read(8 * 1024 * 1024).decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None
    if not text.strip() or text.lstrip()[:15].lower().startswith(
        ("<!doctype", "<html")
    ):
        return None
    parsed = parse_prometheus_text(text)
    parsed["ts"] = round(time.time(), 3)
    return parsed


def _http_get_json(
    url: str, api_key: str | None = None, timeout: float = 2.0
) -> Any | None:
    headers = {"User-Agent": "ai-infra-bench-platform"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener_for_url(url).open(req, timeout=timeout) as resp:
            if not 200 <= resp.status < 300:
                return None
            return json.loads(resp.read(2 * 1024 * 1024).decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def probe_service_info(
    base_url: str, api_key: str | None = None, timeout: float = 2.5
) -> dict[str, Any]:
    """Probe an OpenAI-compatible / SGLang / vLLM endpoint for server metadata."""
    normalized = sanitize_url(base_url)
    info: dict[str, Any] = {
        "base_url": normalized,
        "reachable": False,
        "engine": "openai-compatible",
        "model": None,
        "version": None,
        "tp_size": None,
        "dp_size": None,
        "ep_size": None,
        "quantization": None,
        "kv_cache_dtype": None,
        "speculative_algorithm": None,
        "max_total_tokens": None,
        "server_args": {},
    }

    # 1. SGLang /get_server_info exposes rich server arguments.
    sgl_info = _http_get_json(
        api_url(normalized, "/get_server_info"), api_key=api_key, timeout=timeout
    )
    if isinstance(sgl_info, Mapping):
        info["reachable"] = True
        info["engine"] = "sglang"
        info["version"] = sgl_info.get("version")
        server_args = (
            sgl_info.get("server_args")
            if isinstance(sgl_info.get("server_args"), Mapping)
            else sgl_info
        )
        extracted_args: dict[str, Any] = {}
        keep_keys = (
            "model_path",
            "served_model_name",
            "tp_size",
            "dp_size",
            "ep_size",
            "pp_size",
            "quantization",
            "kv_cache_dtype",
            "attention_backend",
            "speculative_algorithm",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "max_running_requests",
            "max_total_tokens",
            "mem_fraction_static",
            "chunked_prefill_size",
            "cuda_graph_max_bs",
            "enable_dp_attention",
            "enable_cp_attn",
            "enable_cp_attn_decode_tp",
            "enable_metrics",
            "enable_cache_report",
            "disable_radix_cache",
        )
        for key in keep_keys:
            if key in server_args and server_args[key] is not None:
                extracted_args[key] = server_args[key]
        info["server_args"] = extracted_args
        info["model"] = server_args.get("served_model_name") or server_args.get(
            "model_path"
        )
        for key in (
            "tp_size",
            "dp_size",
            "ep_size",
            "quantization",
            "kv_cache_dtype",
            "speculative_algorithm",
            "max_total_tokens",
        ):
            if server_args.get(key) is not None:
                info[key] = server_args.get(key)

    # 2. OpenAI /v1/models (works for SGLang, vLLM, TRT-LLM, etc.)
    models_resp = _http_get_json(
        api_url(normalized, "/v1/models"), api_key=api_key, timeout=timeout
    )
    if isinstance(models_resp, Mapping) and isinstance(models_resp.get("data"), list):
        info["reachable"] = True
        models = models_resp["data"]
        if models and isinstance(models[0], Mapping):
            first = models[0]
            if not info["model"]:
                info["model"] = first.get("id")
            if (
                first.get("owned_by") == "vllm"
                and info["engine"] == "openai-compatible"
            ):
                info["engine"] = "vllm"
            if info["max_total_tokens"] is None and first.get("max_model_len"):
                info["max_total_tokens"] = first.get("max_model_len")

    # 3. vLLM /version
    if info["version"] is None:
        ver_resp = _http_get_json(
            api_url(normalized, "/version"), api_key=api_key, timeout=timeout
        )
        if isinstance(ver_resp, Mapping) and ver_resp.get("version"):
            info["reachable"] = True
            info["version"] = str(ver_resp["version"])
            if info["engine"] == "openai-compatible":
                info["engine"] = "vllm"

    return info


def summarize_telemetry_points(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a sequence of scraped `/metrics` points over a benchmark window."""
    if not points:
        return {"sample_count": 0, "summary": {}, "series": []}

    t0 = float(points[0].get("ts") or 0.0)
    prev_tokens: float | None = None
    prev_ts: float | None = None
    series: list[dict[str, Any]] = []

    kv_vals: list[float] = []
    run_vals: list[int] = []
    wait_vals: list[int] = []
    hit_vals: list[float] = []
    tps_vals: list[float] = []
    spec_vals: list[float] = []

    for pt in points:
        ts = float(pt.get("ts") or t0)
        gen_tps = pt.get("gen_throughput_tps")
        total_toks = pt.get("gen_tokens_total")
        if (
            gen_tps is None
            and total_toks is not None
            and prev_tokens is not None
            and prev_ts is not None
            and ts > prev_ts
        ):
            delta = total_toks - prev_tokens
            if delta >= 0:
                gen_tps = round(delta / (ts - prev_ts), 2)
        if total_toks is not None:
            prev_tokens = float(total_toks)
            prev_ts = ts

        kv = pt.get("kv_cache_pct")
        run_r = pt.get("running_reqs")
        wait_r = pt.get("waiting_reqs")
        hit = pt.get("cache_hit_pct")
        spec_l = pt.get("spec_accept_length")

        if isinstance(kv, (int, float)):
            kv_vals.append(float(kv))
        if isinstance(run_r, (int, float)):
            run_vals.append(int(run_r))
        if isinstance(wait_r, (int, float)):
            wait_vals.append(int(wait_r))
        if isinstance(hit, (int, float)):
            hit_vals.append(float(hit))
        if isinstance(gen_tps, (int, float)):
            tps_vals.append(float(gen_tps))
        if isinstance(spec_l, (int, float)):
            spec_vals.append(float(spec_l))

        series.append(
            {
                "t_rel_s": round(max(0.0, ts - t0), 2),
                "kv_cache_pct": kv,
                "running_reqs": run_r,
                "waiting_reqs": wait_r,
                "gen_throughput_tps": gen_tps,
                "cache_hit_pct": hit,
            }
        )

    summary: dict[str, Any] = {}
    if kv_vals:
        summary["kv_cache_peak_pct"] = round(max(kv_vals), 2)
        summary["kv_cache_mean_pct"] = round(sum(kv_vals) / len(kv_vals), 2)
    if run_vals:
        summary["max_running_reqs"] = max(run_vals)
    if wait_vals:
        summary["max_waiting_reqs"] = max(wait_vals)
    if hit_vals:
        summary["cache_hit_mean_pct"] = round(sum(hit_vals) / len(hit_vals), 2)
    if tps_vals:
        summary["server_gen_tps_peak"] = round(max(tps_vals), 2)
    if spec_vals:
        summary["spec_accept_length_mean"] = round(sum(spec_vals) / len(spec_vals), 2)

    # Downsample series to at most 120 points to keep SQLite rows compact.
    if len(series) > 120:
        step = len(series) / 120.0
        series = [series[min(int(i * step), len(series) - 1)] for i in range(120)]

    return {
        "sample_count": len(points),
        "summary": summary,
        "series": series,
    }


def query_prometheus_range(
    prometheus_url: str,
    query: str,
    start_ts: float,
    end_ts: float,
    step_s: float = 5.0,
    timeout: float = 3.0,
) -> list[tuple[float, float]]:
    """Query Prometheus `/api/v1/query_range` and return `[(unix_ts, float_val)]`."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "start": f"{start_ts:.3f}",
            "end": f"{max(end_ts, start_ts + 0.001):.3f}",
            "step": f"{max(step_s, 1.0):.1f}s",
        }
    )
    url = f"{prometheus_url.rstrip('/')}/api/v1/query_range?{params}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as resp:
            payload = json.loads(resp.read(4 * 1024 * 1024).decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return []
    if payload.get("status") != "success":
        return []
    results = payload.get("data", {}).get("result") or []
    if not results:
        return []
    points: list[tuple[float, float]] = []
    for ts_raw, val_raw in results[0].get("values") or []:
        try:
            v = float(val_raw)
            if math.isfinite(v):
                points.append((float(ts_raw), v))
        except (TypeError, ValueError):
            continue
    return points


def post_grafana_annotation(
    grafana_url: str | None,
    *,
    time_ms: int,
    time_end_ms: int,
    tags: Sequence[str],
    text: str,
    timeout: float = 2.0,
) -> bool:
    """Create a region annotation on the local Grafana instance (best-effort)."""
    if not grafana_url:
        return False
    base = grafana_url.split("/d/")[0].rstrip("/")
    endpoint = f"{base}/api/annotations"
    body = json.dumps(
        {
            "time": time_ms,
            "timeEnd": max(time_end_ms, time_ms + 1),
            "tags": [t for t in tags if t],
            "text": text,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ai-infra-bench-platform",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (OSError, urllib.error.URLError):
        return False


def report_to_platform(
    report_url: str | None,
    payload: Mapping[str, Any],
    *,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Send a CLI benchmark or correctness record to a running AIB platform."""
    if not report_url:
        return None
    endpoint = f"{sanitize_url(report_url)}/api/ingest"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ai-infra-bench-cli",
        },
    )
    try:
        with opener_for_url(endpoint).open(req, timeout=timeout) as resp:
            return json.loads(resp.read(1024 * 1024).decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        logger.warning("Failed to report result to %s: %s", endpoint, error)
        return None
