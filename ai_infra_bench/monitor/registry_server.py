import ipaddress
import json
import logging
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

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
from ai_infra_bench.utils.req import sanitize_url

logger = logging.getLogger(__name__)

MAX_PUSH_BODY_BYTES = 16 * 1024 * 1024
_PUSHED_METRICS_ROUTE = re.compile(r"^/pushed/([^/]+)/metrics$")


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


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class RegistryServer:
    """HTTP control plane for cross-cluster target registration and NAT metric pushes."""

    def __init__(
        self,
        registry: TargetRegistry,
        listen_address: str,
        port: int,
    ):
        self.registry = registry
        self.listen_address = listen_address
        bind_host = listen_address.strip("[]")
        relay_host = _datasource_host(listen_address).strip("[]")
        server_cls = (
            _IPv6ThreadingHTTPServer if ":" in bind_host else ThreadingHTTPServer
        )
        self._server = server_cls((bind_host, port), self._make_handler())
        self.port = self._server.server_port
        self.relay_address = _format_host_port(relay_host, self.port)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aib-monitor-registry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._thread = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, payload: Any) -> None:
                body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(
                self,
                status: int,
                text: str,
                content_type: str = "text/plain; version=0.0.4; charset=utf-8",
            ) -> None:
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.rstrip("/") or "/"

                if route == "/health":
                    self._send_json(200, {"status": "ok"})
                    return
                if route == "/api/targets":
                    targets = outer.registry.snapshot()
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

                self._send_json(404, {"error": f"unknown endpoint: {route}"})

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
                    else:
                        self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except (ConfigurationError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": str(error)})
                except ReloadError as error:
                    self._send_json(500, {"error": str(error)})

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
                        self._send_json(
                            200,
                            {
                                "status": "ok",
                                "removed": [t.label for t in result.removed],
                                "missing": list(result.missing),
                            },
                        )
                    else:
                        self._send_json(404, {"error": f"unknown endpoint: {route}"})
                except (ConfigurationError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(400, {"error": str(error)})
                except ReloadError as error:
                    self._send_json(500, {"error": str(error)})

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

                result = outer.registry.add(
                    urls,
                    labels=raw_labels,
                    metrics_path=metrics_path,
                    target_id=target_id,
                )
                self._send_json(
                    200,
                    {
                        "status": "ok",
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
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "removed": [t.label for t in result.removed],
                        "missing": list(result.missing),
                    },
                )

            def _handle_push(self, query: str) -> None:
                content_type = (self.headers.get("Content-Type") or "").lower()
                params = urllib.parse.parse_qs(query)

                if "application/json" in content_type:
                    body = self._read_json_object()
                    target_id = (
                        body.get("target") or (params.get("target") or [None])[0]
                    )
                    metrics_text = body.get("metrics")
                    base_url = body.get("base_url")
                    raw_labels = body.get("labels")
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
                self._send_json(
                    200,
                    {"status": "ok", "target": _serialize_target(target)},
                )

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
) -> str:
    assigned_id = target_id or _default_push_target_id(target)
    scrape_req = urllib.request.Request(
        target.metrics_url,
        headers={"User-Agent": "ai-infra-bench-monitoring"},
    )
    with _opener_for_url(target.metrics_url).open(scrape_req, timeout=timeout) as resp:
        metrics_text = resp.read(MAX_PUSH_BODY_BYTES).decode("utf-8", errors="replace")

    endpoint = f"{sanitize_url(registry_url)}/api/push"
    _post_json(
        endpoint,
        {
            "target": assigned_id,
            "base_url": target.base_url,
            "labels": target.label_dict,
            "metrics": metrics_text,
        },
        timeout,
    )
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
