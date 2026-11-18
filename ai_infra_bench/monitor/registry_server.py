"""Unified cross-cluster registry, NAT push relay, and Web Studio HTTP server."""

from __future__ import annotations

import ipaddress
import json
import logging
import queue
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ai_infra_bench.monitor.config import (
    DEFAULT_METRICS_PATH,
    ConfigurationError,
    ScrapeTarget,
    _datasource_host,
    _format_host_port,
    build_file_sd_targets,
    parse_labels,
)
from ai_infra_bench.monitor.shell import ReloadError, TargetRegistry
from ai_infra_bench.monitor.telemetry import (
    parse_prometheus_text,
    probe_service_info,
    query_prometheus_range,
    scrape_metrics_point,
)
from ai_infra_bench.utils.req import sanitize_url
from ai_infra_bench.version import __version__

if TYPE_CHECKING:
    from ai_infra_bench.monitor.runner import PlatformRunner
    from ai_infra_bench.monitor.storage import PlatformStore

logger = logging.getLogger(__name__)

MAX_PUSH_BODY_BYTES = 16 * 1024 * 1024
_PUSHED_METRICS_ROUTE = re.compile(r"^/pushed/([^/]+)/metrics$")
_SESSION_DETAIL_ROUTE = re.compile(r"^/api/sessions/([^/]+)$")
_JOB_DETAIL_ROUTE = re.compile(r"^/api/jobs/([^/]+)$")
_JOB_CANCEL_ROUTE = re.compile(r"^/api/jobs/([^/]+)/cancel$")
_EVAL_CASES_ROUTE = re.compile(r"^/api/evals/([^/]+)/cases$")


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _opener_for_url(url: str) -> urllib.request.OpenerDirector:
    parsed = urllib.parse.urlsplit(url)
    if _is_loopback_host(parsed.hostname):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _can_pull_metrics(metrics_url: str, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(
        metrics_url, headers={"User-Agent": "ai-infra-bench-monitoring"}
    )
    try:
        with _opener_for_url(metrics_url).open(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                return False
            sample = response.read(4096).lstrip()
            return bool(sample) and not sample[:15].lower().startswith(
                (b"<!doctype", b"<html")
            )
    except (OSError, urllib.error.URLError):
        return False


def _serialize_target(target: ScrapeTarget) -> dict[str, Any]:
    return {
        "label": target.label,
        "base_url": target.base_url,
        "metrics_url": target.metrics_url,
        "mode": target.mode,
        "labels": target.label_dict,
    }


def _default_push_target_id(target: ScrapeTarget) -> str:
    raw = f"{socket.gethostname()}-{target.address}"
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-")
    return sanitized[:120] or "pushed-0"


def _load_studio_html() -> bytes:
    asset = resources.files("ai_infra_bench.monitor").joinpath("assets", "studio.html")
    return asset.read_text(encoding="utf-8").encode("utf-8")


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True


class _IPv4ThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


class RegistryServer:
    """Unified HTTP control plane for monitoring registration and Benchmark Studio."""

    def __init__(
        self,
        registry: TargetRegistry,
        listen_address: str,
        port: int,
        *,
        store: "PlatformStore | None" = None,
        runner: "PlatformRunner | None" = None,
        prometheus_port: int | None = None,
        grafana_port: int | None = None,
    ):
        self.registry = registry
        self.listen_address = listen_address
        self.prometheus_port = prometheus_port
        self.grafana_port = grafana_port

        self._owns_store = False
        self._owns_runner = False
        if store is None:
            from ai_infra_bench.monitor.storage import PlatformStore

            base_dir = (
                registry.sd_file_path.parent
                if registry.sd_file_path is not None
                else registry.config_path.parent
            )
            store = PlatformStore(Path(base_dir) / "platform.db")
            self._owns_store = True
        self.store = store

        if runner is None:
            from ai_infra_bench.monitor.runner import PlatformRunner

            graf_url = (
                f"http://{_datasource_host(listen_address)}:{grafana_port}"
                if grafana_port is not None
                else None
            )
            runner = PlatformRunner(
                self.store,
                self.registry,
                prometheus_url=self.registry.prometheus_url,
                grafana_url=graf_url,
            )
            self._owns_runner = True
        self.runner = runner

        # Sync any initial static targets into online ServiceSessions.
        for initial_target in self.registry.snapshot():
            self.store.upsert_online_session(
                base_url=initial_target.base_url,
                target_label=initial_target.label,
                metrics_url=initial_target.metrics_url,
                mode=initial_target.mode,
                labels=initial_target.label_dict,
            )

        bind_host = listen_address.strip("[]")
        relay_host = _datasource_host(listen_address).strip("[]")
        server_cls = (
            _IPv6ThreadingHTTPServer if ":" in bind_host else _IPv4ThreadingHTTPServer
        )
        self._server = server_cls((bind_host, port), self._make_handler())
        self.port = self._server.server_port
        self.relay_address = _format_host_port(relay_host, self.port)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._owns_runner:
            self.runner.start()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aib-monitor-registry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            if self._owns_runner:
                self.runner.stop()
            if self._owns_store:
                self.store.close()
            return
        self.runner.events.close_all()
        if self._owns_runner:
            self.runner.stop()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._thread = None
        if self._owns_store:
            self.store.close()

    def _sync_expired_push_sessions(
        self, active_targets: Sequence[ScrapeTarget]
    ) -> None:
        active_labels = {t.label for t in active_targets}
        online_sessions = self.store.list_sessions(status="online")
        stale = [
            str(s["target_label"])
            for s in online_sessions
            if s.get("mode") == "push"
            and s.get("target_label")
            and s["target_label"] not in active_labels
        ]
        if stale:
            self.store.mark_targets_offline(stale)

    def _probe_and_enrich_async(self, session_id: str, base_url: str) -> None:
        def _bg() -> None:
            if base_url.startswith("push://"):
                return
            info = probe_service_info(base_url, timeout=2.0)
            if info.get("reachable"):
                updated = self.store.update_session(session_id, server_info=info)
                if updated:
                    self.runner.events.publish("session_updated", updated)

        threading.Thread(target=_bg, daemon=True).start()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, payload: Any) -> None:
                body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(
                self,
                status: int,
                text: str,
                content_type: str = "text/plain; version=0.0.4; charset=utf-8",
            ) -> None:
                self._send_bytes(status, text.encode("utf-8"), content_type)

            def _read_body(self) -> bytes:
                raw_len = self.headers.get("Content-Length")
                if raw_len is None:
                    raise ValueError("Content-Length header is required")
                length = int(raw_len)
                if length < 0 or length > MAX_PUSH_BODY_BYTES:
                    raise ValueError(f"payload size {length} exceeds limit")
                return self.rfile.read(length)

            def _read_json_object(self) -> Mapping[str, Any]:
                raw = self._read_body()
                if not raw.strip():
                    return {}
                parsed = json.loads(raw.decode("utf-8"))
                if not isinstance(parsed, Mapping):
                    raise ValueError("request body must be a JSON object")
                return parsed

            # -------------------- GET --------------------
            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.rstrip("/") or "/"
                params = urllib.parse.parse_qs(parsed.query)

                try:
                    if route in {"/", "/studio"}:
                        self._send_bytes(
                            200, _load_studio_html(), "text/html; charset=utf-8"
                        )
                        return
                    if route == "/health":
                        self._send_json(200, {"status": "ok"})
                        return
                    if route == "/api/platform/info":
                        self._send_json(
                            200,
                            {
                                "status": "ok",
                                "version": __version__,
                                "registry_port": outer.port,
                                "prometheus_port": outer.prometheus_port,
                                "grafana_port": outer.grafana_port,
                                "prometheus_enabled": outer.prometheus_port is not None,
                                "grafana_enabled": outer.grafana_port is not None,
                            },
                        )
                        return
                    if route == "/api/targets":
                        targets = outer.registry.snapshot()
                        outer._sync_expired_push_sessions(targets)
                        self._send_json(
                            200,
                            {"targets": [_serialize_target(t) for t in targets]},
                        )
                        return
                    if route == "/api/sd/prometheus":
                        self._send_json(
                            200, build_file_sd_targets(outer.registry.snapshot())
                        )
                        return
                    if match := _PUSHED_METRICS_ROUTE.fullmatch(route):
                        target_id = urllib.parse.unquote(match.group(1))
                        metrics_text = outer.registry.get_pushed_metrics(target_id)
                        if metrics_text is None:
                            self._send_text(
                                404,
                                "target expired or not found\n",
                                content_type="text/plain; charset=utf-8",
                            )
                        else:
                            self._send_text(200, metrics_text)
                        return
                    if route == "/api/monitor/live":
                        self._handle_monitor_live()
                        return
                    if route == "/api/monitor/timeseries":
                        self._handle_monitor_timeseries(params)
                        return
                    if route == "/api/sessions":
                        status_filter = (params.get("status") or [None])[0]
                        self._send_json(
                            200,
                            {
                                "sessions": outer.store.list_sessions(
                                    status=status_filter
                                )
                            },
                        )
                        return
                    if match := _SESSION_DETAIL_ROUTE.fullmatch(route):
                        sid = urllib.parse.unquote(match.group(1))
                        detail = outer.store.get_session_detail(sid)
                        if detail is None:
                            self._send_json(
                                404, {"error": f"session {sid!r} not found"}
                            )
                        else:
                            self._send_json(200, detail)
                        return
                    if route == "/api/jobs":
                        sid = (params.get("session_id") or [None])[0]
                        self._send_json(
                            200, {"jobs": outer.store.list_jobs(session_id=sid)}
                        )
                        return
                    if match := _JOB_DETAIL_ROUTE.fullmatch(route):
                        jid = urllib.parse.unquote(match.group(1))
                        job = outer.store.get_job(jid)
                        if job is None:
                            self._send_json(404, {"error": f"job {jid!r} not found"})
                        else:
                            self._send_json(200, job)
                        return
                    if route == "/api/compare":
                        raw_ids = (params.get("sessions") or [""])[0]
                        ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
                        self._send_json(200, outer.store.compare_sessions(ids))
                        return
                    if match := _EVAL_CASES_ROUTE.fullmatch(route):
                        eid = urllib.parse.unquote(match.group(1))
                        cases_doc = outer.store.get_eval_cases(eid)
                        if cases_doc is None:
                            self._send_json(404, {"error": f"eval {eid!r} not found"})
                        else:
                            self._send_json(200, cases_doc)
                        return
                    if route == "/api/events":
                        self._handle_sse()
                        return

                    self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except Exception as error:
                    logger.exception("GET %s failed", route)
                    self._send_json(500, {"error": str(error)})

            # -------------------- POST --------------------
            def do_POST(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.rstrip("/") or "/"
                try:
                    if route == "/api/register":
                        self._handle_register()
                    elif route == "/api/deregister":
                        self._handle_deregister(parsed.query)
                    elif route == "/api/push":
                        self._handle_push(parsed.query)
                    elif route == "/api/probe":
                        self._handle_probe()
                    elif route == "/api/sessions":
                        self._handle_create_session()
                    elif match := _SESSION_DETAIL_ROUTE.fullmatch(route):
                        self._handle_update_session(
                            urllib.parse.unquote(match.group(1))
                        )
                    elif route == "/api/jobs":
                        self._handle_create_job()
                    elif match := _JOB_CANCEL_ROUTE.fullmatch(route):
                        jid = urllib.parse.unquote(match.group(1))
                        job = outer.runner.cancel_job(jid)
                        if job is None:
                            self._send_json(404, {"error": f"job {jid!r} not found"})
                        else:
                            self._send_json(200, job)
                    elif route == "/api/ingest":
                        self._handle_ingest()
                    else:
                        self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except (ConfigurationError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": str(error)})
                except ReloadError as error:
                    self._send_json(500, {"error": str(error)})

            # -------------------- PATCH --------------------
            def do_PATCH(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.rstrip("/") or "/"
                try:
                    if match := _SESSION_DETAIL_ROUTE.fullmatch(route):
                        self._handle_update_session(
                            urllib.parse.unquote(match.group(1))
                        )
                    else:
                        self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except (ConfigurationError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": str(error)})

            # -------------------- DELETE --------------------
            def do_DELETE(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.rstrip("/") or "/"
                try:
                    if route in {"/api/register", "/api/targets"}:
                        self._handle_deregister(parsed.query)
                    elif route.startswith("/api/targets/"):
                        ident = urllib.parse.unquote(
                            route.removeprefix("/api/targets/")
                        )
                        result = outer.registry.delete([ident])
                        removed_labels = [t.label for t in result.removed]
                        outer.store.mark_targets_offline(removed_labels)
                        outer.runner.events.publish(
                            "target_changed", {"removed": removed_labels}
                        )
                        self._send_json(
                            200,
                            {
                                "status": "ok",
                                "removed": removed_labels,
                                "missing": list(result.missing),
                            },
                        )
                    elif match := _SESSION_DETAIL_ROUTE.fullmatch(route):
                        sid = urllib.parse.unquote(match.group(1))
                        deleted = outer.store.delete_session(sid)
                        if not deleted:
                            self._send_json(
                                404, {"error": f"session {sid!r} not found"}
                            )
                        else:
                            self._send_json(200, {"status": "ok", "deleted": sid})
                    else:
                        self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except (ConfigurationError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": str(error)})
                except ReloadError as error:
                    self._send_json(500, {"error": str(error)})

            # -------------------- Handlers --------------------
            def _handle_sse(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                sub = outer.runner.events.subscribe()
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    while True:
                        try:
                            msg = sub.get(timeout=15.0)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        if msg is None:
                            break
                        encoded = json.dumps(msg, ensure_ascii=False)
                        frame = f"event: {msg['event']}\ndata: {encoded}\n\n".encode(
                            "utf-8"
                        )
                        self.wfile.write(frame)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    outer.runner.events.unsubscribe(sub)

            def _handle_monitor_live(self) -> None:
                targets = outer.registry.snapshot()
                outer._sync_expired_push_sessions(targets)
                items = []
                for t in targets:
                    session = outer.store.get_online_session_for_target(t.label)
                    if session is None:
                        session = outer.store.upsert_online_session(
                            base_url=t.base_url,
                            target_label=t.label,
                            metrics_url=t.metrics_url,
                            mode=t.mode,
                            labels=t.label_dict,
                        )
                    latest = outer.store.get_latest_point(t.label)
                    if latest is None and t.mode == "pull":
                        scraped = scrape_metrics_point(t.metrics_url, timeout=1.2)
                        if scraped is not None:
                            latest = outer.store.record_live_point(t.label, scraped)
                    series = outer.store.get_live_series(t.label, limit=40)
                    items.append(
                        {
                            **_serialize_target(t),
                            "session_id": session["session_id"],
                            "session_name": session["name"],
                            "server_info": session["server_info"],
                            "hardware_info": session["hardware_info"],
                            "latest": latest or {},
                            "series": series,
                        }
                    )
                self._send_json(200, {"targets": items})

            def _handle_monitor_timeseries(
                self, params: Mapping[str, list[str]]
            ) -> None:
                target_lbl = (params.get("target") or [""])[0].strip()
                minutes = max(1, min(180, int((params.get("minutes") or ["15"])[0])))
                if not target_lbl:
                    self._send_json(400, {"error": "missing 'target' query param"})
                    return
                series = outer.store.get_live_series(target_lbl, limit=90)
                if not series and outer.prometheus_port is not None:
                    import time

                    end_ts = time.time()
                    start_ts = end_ts - minutes * 60
                    prom_url = outer.registry.prometheus_url
                    kv_pts = query_prometheus_range(
                        prom_url,
                        f'sglang:token_usage{{aib_target="{target_lbl}"}} * 100 '
                        f'or vllm:gpu_cache_usage_perc{{aib_target="{target_lbl}"}} * 100',
                        start_ts,
                        end_ts,
                        step_s=max(5.0, minutes * 60 / 90.0),
                    )
                    tps_pts = query_prometheus_range(
                        prom_url,
                        f'sglang:gen_throughput{{aib_target="{target_lbl}"}}',
                        start_ts,
                        end_ts,
                        step_s=max(5.0, minutes * 60 / 90.0),
                    )
                    tps_map = {round(ts, 1): val for ts, val in tps_pts}
                    series = [
                        {
                            "ts": ts,
                            "kv_cache_pct": round(val, 2),
                            "gen_throughput_tps": tps_map.get(round(ts, 1)),
                        }
                        for ts, val in kv_pts
                    ]
                self._send_json(200, {"target": target_lbl, "series": series})

            def _handle_register(self) -> None:
                body = self._read_json_object()
                urls: list[str] = []
                for single_key in ("url", "base_url"):
                    if val := body.get(single_key):
                        if not isinstance(val, str):
                            raise ValueError(f"{single_key!r} must be a string")
                        urls.append(val)
                for list_key in ("urls", "base_urls"):
                    if vals := body.get(list_key):
                        if not isinstance(vals, list) or not all(
                            isinstance(x, str) for x in vals
                        ):
                            raise ValueError(f"{list_key!r} must be a list of strings")
                        urls.extend(vals)
                if not urls:
                    raise ValueError("missing 'url' or 'urls' in registration payload")

                raw_labels = body.get("labels")
                if raw_labels is not None and not isinstance(
                    raw_labels, (Mapping, list)
                ):
                    raise ValueError("'labels' must be an object or KEY=VALUE list")
                metrics_path = body.get("metrics_path")
                if metrics_path is not None and not isinstance(metrics_path, str):
                    raise ValueError("'metrics_path' must be a string")
                target_id = body.get("target")
                if target_id is not None and not isinstance(target_id, str):
                    raise ValueError("'target' must be a string")
                verify_pull = bool(body.get("verify_pull", False))
                srv_info = (
                    body.get("server_info")
                    if isinstance(body.get("server_info"), Mapping)
                    else None
                )
                hw_info = (
                    body.get("hardware_info")
                    if isinstance(body.get("hardware_info"), Mapping)
                    else None
                )

                if verify_pull:
                    from ai_infra_bench.monitor.config import parse_targets

                    candidates = parse_targets(
                        urls, metrics_path or outer.registry.metrics_path
                    )
                    if not all(
                        _can_pull_metrics(c.metrics_url, timeout=2.0)
                        for c in candidates
                    ):
                        self._send_json(
                            200,
                            {
                                "status": "ok",
                                "mode": "push_required",
                                "added": [],
                                "existing": [],
                            },
                        )
                        return

                result = outer.registry.add(
                    urls,
                    labels=raw_labels,
                    metrics_path=metrics_path,
                    target_id=target_id,
                )
                for item in (*result.added, *result.existing):
                    sess = outer.store.upsert_online_session(
                        base_url=item.base_url,
                        target_label=item.label,
                        metrics_url=item.metrics_url,
                        mode="pull",
                        labels=item.label_dict,
                        server_info=srv_info,
                        hardware_info=hw_info,
                    )
                    if not srv_info:
                        outer._probe_and_enrich_async(sess["session_id"], item.base_url)

                outer.runner.events.publish(
                    "target_changed",
                    {"added": [t.label for t in result.added]},
                )
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "mode": "pull",
                        "added": [_serialize_target(t) for t in result.added],
                        "existing": [_serialize_target(t) for t in result.existing],
                    },
                )

            def _handle_deregister(self, query: str) -> None:
                params = urllib.parse.parse_qs(query)
                identifiers: list[str] = []
                for key in ("target", "url"):
                    identifiers.extend(params.get(key, []))

                if self.headers.get("Content-Length") not in (None, "0"):
                    body = self._read_json_object()
                    for single_key in ("target", "url", "base_url"):
                        if val := body.get(single_key):
                            if not isinstance(val, str):
                                raise ValueError(f"{single_key!r} must be a string")
                            identifiers.append(val)
                    for list_key in ("targets", "urls"):
                        if vals := body.get(list_key):
                            if not isinstance(vals, list) or not all(
                                isinstance(x, str) for x in vals
                            ):
                                raise ValueError(
                                    f"{list_key!r} must be a list of strings"
                                )
                            identifiers.extend(vals)

                if not identifiers:
                    raise ValueError("missing 'target' or 'url' to deregister")

                result = outer.registry.delete(identifiers)
                removed_labels = [t.label for t in result.removed]
                outer.store.mark_targets_offline(removed_labels)
                outer.runner.events.publish(
                    "target_changed", {"removed": removed_labels}
                )
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "removed": removed_labels,
                        "missing": list(result.missing),
                    },
                )

            def _handle_push(self, query: str) -> None:
                content_type = (self.headers.get("Content-Type") or "").lower()
                params = urllib.parse.parse_qs(query)
                srv_info = None
                hw_info = None

                if "application/json" in content_type:
                    body = self._read_json_object()
                    target_id = (
                        body.get("target") or (params.get("target") or [None])[0]
                    )
                    metrics_text = body.get("metrics")
                    base_url = body.get("base_url")
                    raw_labels = body.get("labels")
                    srv_info = (
                        body.get("server_info")
                        if isinstance(body.get("server_info"), Mapping)
                        else None
                    )
                    hw_info = (
                        body.get("hardware_info")
                        if isinstance(body.get("hardware_info"), Mapping)
                        else None
                    )
                    if not isinstance(target_id, str) or not target_id.strip():
                        raise ValueError("'target' is required when pushing metrics")
                    if not isinstance(metrics_text, str):
                        raise ValueError("'metrics' string is required")
                    if base_url is not None and not isinstance(base_url, str):
                        raise ValueError("'base_url' must be a string")
                    if raw_labels is not None and not isinstance(
                        raw_labels, (Mapping, list)
                    ):
                        raise ValueError("'labels' must be an object or KEY=VALUE list")
                    labels = parse_labels(raw_labels)
                else:
                    target_id = (params.get("target") or [None])[0]
                    if not target_id:
                        raise ValueError(
                            "missing '?target=<id>' query parameter for raw metrics push"
                        )
                    base_url = (params.get("base_url") or [None])[0]
                    query_labels = {
                        k: vals[-1]
                        for k, vals in params.items()
                        if k not in {"target", "base_url"} and vals
                    }
                    labels = parse_labels(query_labels)
                    metrics_text = self._read_body().decode("utf-8")

                target = outer.registry.upsert_pushed(
                    target_id,
                    outer.relay_address,
                    metrics_text,
                    labels=labels,
                    base_url=base_url,
                )
                parsed_pt = parse_prometheus_text(metrics_text)
                enriched = outer.store.record_live_point(target.label, parsed_pt)
                outer.store.upsert_online_session(
                    base_url=target.base_url,
                    target_label=target.label,
                    metrics_url=target.metrics_url,
                    mode="push",
                    labels=target.label_dict,
                    server_info=srv_info,
                    hardware_info=hw_info,
                )
                outer.runner.events.publish(
                    "monitor_tick", {"target": target.label, "point": enriched}
                )
                self._send_json(
                    200,
                    {"status": "ok", "target": _serialize_target(target)},
                )

            def _handle_probe(self) -> None:
                body = self._read_json_object()
                raw_url = body.get("base_url") or body.get("url")
                if not isinstance(raw_url, str) or not raw_url.strip():
                    raise ValueError("'base_url' is required")
                api_key = body.get("api_key")
                info = probe_service_info(
                    raw_url, api_key=str(api_key) if api_key else None
                )
                metrics_pt = scrape_metrics_point(
                    f"{sanitize_url(raw_url)}/metrics", timeout=1.5
                )
                self._send_json(
                    200,
                    {
                        "server_info": info,
                        "metrics_available": metrics_pt is not None,
                        "latest_metrics": metrics_pt,
                    },
                )

            def _handle_create_session(self) -> None:
                body = self._read_json_object()
                raw_url = body.get("base_url") or body.get("url")
                if not isinstance(raw_url, str) or not raw_url.strip():
                    raise ValueError("'base_url' is required")
                name = body.get("name")
                force_new = bool(body.get("force_new", False))
                labels = (
                    parse_labels(body.get("labels"))
                    if body.get("labels") is not None
                    else ()
                )
                info = probe_service_info(raw_url, timeout=2.0)
                sess = outer.store.upsert_online_session(
                    base_url=raw_url,
                    target_label=body.get("target_label"),
                    mode="external",
                    labels=dict(labels),
                    server_info=info if info.get("reachable") else {},
                    name=str(name).strip() if name else None,
                    force_new=force_new,
                )
                self._send_json(200, sess)

            def _handle_update_session(self, session_id: str) -> None:
                body = self._read_json_object()
                updated = outer.store.update_session(
                    session_id,
                    name=body.get("name"),
                    notes=body.get("notes"),
                    labels=body.get("labels"),
                    is_baseline=body.get("is_baseline"),
                )
                if updated is None:
                    self._send_json(404, {"error": f"session {session_id!r} not found"})
                else:
                    outer.runner.events.publish("session_updated", updated)
                    self._send_json(200, updated)

            def _handle_create_job(self) -> None:
                body = self._read_json_object()
                job_type = body.get("job_type")
                if not isinstance(job_type, str) or not job_type:
                    raise ValueError("'job_type' is required")
                params = body.get("params") or {}
                if not isinstance(params, Mapping):
                    raise ValueError("'params' must be a JSON object")

                session_id = body.get("session_id")
                target_label = body.get("target") or body.get("target_label")
                raw_url = body.get("base_url")
                session_name = body.get("session_name")
                force_new_session = bool(body.get("force_new_session", False))

                if not session_id:
                    if target_label:
                        existing = outer.store.get_online_session_for_target(
                            str(target_label)
                        )
                        if existing and not force_new_session:
                            session_id = existing["session_id"]
                        else:
                            tgt = next(
                                (
                                    t
                                    for t in outer.registry.snapshot()
                                    if t.label == target_label
                                ),
                                None,
                            )
                            if tgt is None and not raw_url:
                                raise ValueError(
                                    f"target {target_label!r} not found; supply base_url"
                                )
                            created = outer.store.upsert_online_session(
                                base_url=raw_url or tgt.base_url,  # type: ignore[union-attr]
                                target_label=tgt.label if tgt else str(target_label),
                                metrics_url=tgt.metrics_url if tgt else None,
                                mode=tgt.mode if tgt else "pull",
                                labels=tgt.label_dict if tgt else {},
                                name=session_name,
                                force_new=force_new_session,
                            )
                            session_id = created["session_id"]
                    elif isinstance(raw_url, str) and raw_url.strip():
                        info = probe_service_info(raw_url, timeout=2.0)
                        created = outer.store.upsert_online_session(
                            base_url=raw_url,
                            mode="external",
                            server_info=info if info.get("reachable") else {},
                            name=session_name,
                            force_new=force_new_session,
                        )
                        session_id = created["session_id"]
                    else:
                        raise ValueError(
                            "one of 'session_id', 'target', or 'base_url' is required"
                        )

                if session_name and not force_new_session:
                    outer.store.update_session(str(session_id), name=str(session_name))

                job = outer.runner.submit_job(str(session_id), job_type, params)
                self._send_json(200, job)

            def _handle_ingest(self) -> None:
                body = self._read_json_object()
                kind = str(body.get("kind") or "")
                raw_url = str(body.get("base_url") or "")
                if not raw_url:
                    raise ValueError("'base_url' is required in ingest payload")

                info = probe_service_info(raw_url, timeout=1.5)
                sess = outer.store.upsert_online_session(
                    base_url=raw_url,
                    labels=(
                        {"model": str(body["model"])} if body.get("model") else None
                    ),
                    server_info=info if info.get("reachable") else None,
                )
                sid = sess["session_id"]

                if kind == "bench":
                    record = body.get("record") or {}
                    saved = outer.store.add_bench_run(
                        session_id=sid,
                        subcommand=str(body.get("subcommand") or "bench"),
                        workload=str(body.get("workload") or ""),
                        yaml_record=record,
                        t_start=body.get("t_start"),
                        t_end=body.get("t_end"),
                    )
                    outer.runner.events.publish("bench_run_added", saved)
                    self._send_json(
                        200, {"status": "ok", "session_id": sid, "run": saved}
                    )
                    return

                if kind == "correctness":
                    saved_list = []
                    for rec in body.get("runs") or []:
                        saved = outer.store.add_correctness_run(
                            session_id=sid,
                            eval_type="dataset",
                            suite_name=str(rec.get("suite_name") or "dataset"),
                            passed=bool(rec.get("passed", True)),
                            correct_rate=rec.get("correct_rate"),
                            wrong_rate=rec.get("wrong_rate"),
                            failed_rate=rec.get("failed_rate"),
                            truncated_rate=rec.get("truncated_rate"),
                            summary=rec.get("summary") or {},
                            cases=rec.get("cases"),
                            t_start=rec.get("t_start"),
                            t_end=rec.get("t_end"),
                        )
                        saved_list.append(saved)
                    self._send_json(
                        200, {"status": "ok", "session_id": sid, "evals": saved_list}
                    )
                    return

                if kind == "logits":
                    res = dict(body.get("result") or {})
                    positions = res.pop("positions", [])
                    saved = outer.store.add_correctness_run(
                        session_id=sid,
                        eval_type="logits",
                        suite_name="eval-logits",
                        passed=bool(res.get("passed")),
                        correct_rate=res.get("token_match_rate"),
                        ref_base_url=body.get("ref_base_url"),
                        summary=res,
                        cases=positions,
                    )
                    self._send_json(
                        200, {"status": "ok", "session_id": sid, "eval": saved}
                    )
                    return

                if kind == "hidden_states":
                    res = dict(body.get("result") or {})
                    saved = outer.store.add_correctness_run(
                        session_id=sid,
                        eval_type="hidden_states",
                        suite_name="eval-hidden-states",
                        passed=bool(res.get("passed")),
                        correct_rate=res.get("close_ratio"),
                        ref_base_url=body.get("ref_base_url"),
                        summary=res,
                    )
                    self._send_json(
                        200, {"status": "ok", "session_id": sid, "eval": saved}
                    )
                    return

                raise ValueError(f"unsupported ingest kind: {kind!r}")

            def log_message(self, *_args: Any) -> None:
                return None

        return Handler


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ai-infra-bench-monitoring",
        },
    )
    try:
        with _opener_for_url(url).open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace").strip()
        raise ReloadError(
            f"registry request to {url} failed (HTTP {error.code}): {detail}"
        ) from error
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise ReloadError(f"registry request to {url} failed: {error}") from error


def register_remote_targets(
    registry_url: str,
    base_urls: Sequence[str],
    *,
    metrics_path: str = DEFAULT_METRICS_PATH,
    labels: tuple[tuple[str, str], ...] = (),
    timeout: float = 5.0,
) -> dict[str, Any]:
    endpoint = f"{sanitize_url(registry_url)}/api/register"
    payload: dict[str, Any] = {
        "urls": list(base_urls),
        "metrics_path": metrics_path,
    }
    if labels:
        payload["labels"] = dict(labels)
    return _post_json(endpoint, payload, timeout)


def deregister_remote_targets(
    registry_url: str,
    identifiers: Sequence[str],
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    endpoint = f"{sanitize_url(registry_url)}/api/deregister"
    return _post_json(endpoint, {"targets": list(identifiers)}, timeout)


def push_metrics_once(
    registry_url: str,
    target: ScrapeTarget,
    *,
    target_id: str | None = None,
    timeout: float = 5.0,
    base_url_override: str | None = None,
    server_info: Mapping[str, Any] | None = None,
    hardware_info: Mapping[str, Any] | None = None,
) -> str:
    assigned_id = target_id or _default_push_target_id(target)
    scrape_req = urllib.request.Request(
        target.metrics_url,
        headers={"User-Agent": "ai-infra-bench-monitoring"},
    )
    with _opener_for_url(target.metrics_url).open(scrape_req, timeout=timeout) as resp:
        metrics_text = resp.read(MAX_PUSH_BODY_BYTES).decode("utf-8", errors="replace")

    endpoint = f"{sanitize_url(registry_url)}/api/push"
    body: dict[str, Any] = {
        "target": assigned_id,
        "base_url": base_url_override or target.base_url,
        "labels": target.label_dict,
        "metrics": metrics_text,
    }
    if server_info is not None:
        body["server_info"] = dict(server_info)
    if hardware_info is not None:
        body["hardware_info"] = dict(hardware_info)
    _post_json(endpoint, body, timeout)
    return assigned_id


def run_push_agent(
    registry_url: str,
    targets: Sequence[ScrapeTarget],
    interval_s: float,
    *,
    target_timeout: float = 5.0,
    stop_event: threading.Event | None = None,
) -> int:
    stopper = stop_event or threading.Event()
    target_ids = [
        (
            f"{_default_push_target_id(t)}-{idx}"
            if len(targets) > 1
            else _default_push_target_id(t)
        )
        for idx, t in enumerate(targets)
    ]
    normalized_registry = sanitize_url(registry_url)
    print(f"Push registry: {normalized_registry}")
    for tid, target in zip(target_ids, targets):
        print(f"  {tid}: {target.metrics_url} -> {normalized_registry}/api/push")

    registered_ids: set[str] = set()
    try:
        while not stopper.is_set():
            for tid, target in zip(target_ids, targets):
                try:
                    push_metrics_once(
                        registry_url,
                        target,
                        target_id=tid,
                        timeout=target_timeout,
                    )
                    registered_ids.add(tid)
                except Exception as error:
                    logger.warning(
                        "Failed to push metrics for %s (%s): %s",
                        tid,
                        target.metrics_url,
                        error,
                    )
            stopper.wait(interval_s)
    except KeyboardInterrupt:
        print("\nStopping metrics push agent...")
    finally:
        if registered_ids:
            try:
                deregister_remote_targets(
                    registry_url, sorted(registered_ids), timeout=target_timeout
                )
            except Exception as error:
                logger.debug("Best-effort deregister on shutdown failed: %s", error)
    return 0
