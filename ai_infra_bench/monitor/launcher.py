"""Zero-intrusion service wrapper and embedded auto-registration helper."""

import atexit
import ipaddress
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ai_infra_bench.monitor.config import (
    DEFAULT_METRICS_PATH,
    _format_host_port,
    _normalize_metrics_path,
    parse_duration_seconds,
    parse_labels,
    parse_targets,
)
from ai_infra_bench.monitor.registry_server import (
    _default_push_target_id,
    _opener_for_url,
    _post_json,
    deregister_remote_targets,
    push_metrics_once,
)
from ai_infra_bench.utils.req import sanitize_url

logger = logging.getLogger(__name__)


def infer_command_metadata(
    command: Sequence[str],
) -> tuple[int, str, dict[str, str]]:
    """Infer listening port, metrics path, and model name from a launch command."""
    joined = " ".join(command).lower()
    port = 8000 if "vllm" in joined and "sglang" not in joined else 30000
    metrics_path = DEFAULT_METRICS_PATH
    model_name: str | None = None

    tokens = list(command)
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        key, eq, inline_val = tok.partition("=")
        next_val = (
            inline_val if eq else (tokens[idx + 1] if idx + 1 < len(tokens) else None)
        )
        step = 1 if eq else 2

        if key in {"--port", "-p", "--prometheus-port"} and next_val:
            try:
                port = int(next_val)
                idx += step
                continue
            except ValueError:
                pass
        if key in {"--served-model-name", "--model-path", "--model"} and next_val:
            # Prefer --served-model-name if seen; otherwise take basename of path/repo.
            candidate = Path(next_val.rstrip("/")).name or next_val
            if key == "--served-model-name" or model_name is None:
                model_name = candidate
            idx += step
            continue
        if tok == "serve" and idx > 0 and "vllm" in tokens[idx - 1]:
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                model_name = model_name or (
                    Path(tokens[idx + 1].rstrip("/")).name or tokens[idx + 1]
                )
        idx += 1

    inferred_labels = {"model": model_name} if model_name else {}
    return port, metrics_path, inferred_labels


def detect_advertise_host(monitor_url: str) -> str:
    """Discover the local IP address routable towards ``monitor_url``."""
    if raw_override := os.environ.get("AIB_ADVERTISE_HOST", "").strip():
        override_host = urllib.parse.urlsplit(sanitize_url(raw_override)).hostname
        if override_host:
            return override_host

    parsed = urllib.parse.urlsplit(sanitize_url(monitor_url))
    target_host = parsed.hostname or "127.0.0.1"
    target_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if target_host in {"localhost", "127.0.0.1", "::1"}:
        return "127.0.0.1"

    if pod_ip := os.environ.get("POD_IP", "").strip():
        try:
            ipaddress.ip_address(pod_ip)
            return pod_ip
        except ValueError:
            pass

    try:
        infos = socket.getaddrinfo(target_host, target_port, type=socket.SOCK_DGRAM)
        family, _, _, _, sockaddr = infos[0]
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.connect(sockaddr)
            detected = sock.getsockname()[0]
            if detected and not detected.startswith("0."):
                return detected
    except OSError:
        pass
    return "127.0.0.1"


def _local_metrics_ready(url: str, timeout: float = 1.5) -> bool:
    req = urllib.request.Request(
        url, headers={"User-Agent": "ai-infra-bench-monitoring"}
    )
    try:
        with _opener_for_url(url).open(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (OSError, urllib.error.URLError):
        return False


def auto_register(
    monitor_url: str | None = None,
    *,
    port: int = 30000,
    advertise_host: str | None = None,
    metrics_path: str = DEFAULT_METRICS_PATH,
    labels: Sequence[str] | Mapping[str, str] | None = None,
    scrape_interval_s: float = 5.0,
    ready_timeout_s: float = 900.0,
    should_abort: Callable[[], bool] | None = None,
) -> Callable[[], None]:
    """Wait for local ``/metrics`` in a background thread, then auto-register.

    Automatically negotiates direct Pull registration when the monitor server can
    reach ``advertise_host:port``, or falls back to periodic NAT Push when isolated.
    Returns an idempotent ``close()`` callback that deregisters on shutdown.
    """
    resolved_monitor = (monitor_url or os.environ.get("AIB_MONITOR_URL") or "").strip()
    if not resolved_monitor:
        return lambda: None

    env_labels = dict(
        parse_labels(
            [os.environ["AIB_MONITOR_LABELS"]]
            if os.environ.get("AIB_MONITOR_LABELS")
            else None
        )
    )
    explicit_labels = dict(parse_labels(labels))
    merged_labels = parse_labels({**env_labels, **explicit_labels})

    normalized_path = _normalize_metrics_path(metrics_path)
    host = advertise_host or detect_advertise_host(resolved_monitor)
    advertise_url = f"http://{_format_host_port(host, port)}"
    local_scrape_target = parse_targets(
        [f"http://127.0.0.1:{port}"], normalized_path, labels=merged_labels
    )[0]

    stop_event = threading.Event()
    registered_identifiers: list[str] = []
    closed_lock = threading.Lock()
    closed = False

    def _worker() -> None:
        deadline = time.monotonic() + ready_timeout_s
        while not stop_event.is_set():
            if should_abort is not None and should_abort():
                return
            if _local_metrics_ready(local_scrape_target.metrics_url):
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "Timed out waiting for local metrics at %s; skipping auto-register",
                    local_scrape_target.metrics_url,
                )
                return
            stop_event.wait(0.5)

        if stop_event.is_set():
            return

        reg_endpoint = f"{sanitize_url(resolved_monitor)}/api/register"
        try:
            response = _post_json(
                reg_endpoint,
                {
                    "url": advertise_url,
                    "metrics_path": normalized_path,
                    "labels": dict(merged_labels),
                    "verify_pull": True,
                },
                timeout=5.0,
            )
            mode = response.get("mode", "pull")
            if mode == "pull":
                for item in (
                    *response.get("added", []),
                    *response.get("existing", []),
                ):
                    if lbl := item.get("label"):
                        registered_identifiers.append(lbl)
                if not registered_identifiers:
                    registered_identifiers.append(advertise_url)
                logger.info(
                    "Auto-registered %s with monitor %s (pull mode)",
                    advertise_url,
                    resolved_monitor,
                )
                return
        except Exception as error:
            logger.warning(
                "Pull registration with %s failed (%s); falling back to push mode",
                resolved_monitor,
                error,
            )

        # NAT-isolated fallback: push local /metrics periodically.
        target_id = _default_push_target_id(
            parse_targets([advertise_url], normalized_path, labels=merged_labels)[0]
        )
        logger.info(
            "Direct pull unreachable from %s; pushing local metrics as %s",
            resolved_monitor,
            target_id,
        )
        while not stop_event.is_set():
            if should_abort is not None and should_abort():
                return
            try:
                push_metrics_once(
                    resolved_monitor,
                    local_scrape_target,
                    target_id=target_id,
                    timeout=5.0,
                )
                if target_id not in registered_identifiers:
                    registered_identifiers.append(target_id)
            except Exception as error:
                logger.warning("Auto-push to %s failed: %s", resolved_monitor, error)
            stop_event.wait(max(scrape_interval_s, 0.2))

    thread = threading.Thread(
        target=_worker, name="aib-monitor-autoregister", daemon=True
    )
    thread.start()

    def close() -> None:
        nonlocal closed
        with closed_lock:
            if closed:
                return
            closed = True
        stop_event.set()
        thread.join(timeout=2.0)
        if registered_identifiers:
            try:
                deregister_remote_targets(
                    resolved_monitor, list(registered_identifiers), timeout=3.0
                )
            except Exception as error:
                logger.debug("Auto-deregister skipped: %s", error)

    atexit.register(close)
    return close


def run_wrapped_service(
    command: Sequence[str],
    monitor_url: str,
    *,
    base_urls: Sequence[str] = (),
    labels: tuple[tuple[str, str], ...] = (),
    metrics_path: str | None = None,
    scrape_interval: str = "5s",
) -> int:
    """Launch ``command``, auto-register once healthy, and deregister on exit."""
    inferred_port, inferred_path, inferred_labels = infer_command_metadata(command)
    merged_label_map = {**inferred_labels, **dict(labels)}
    effective_path = metrics_path or inferred_path
    advertise_host: str | None = None

    if base_urls:
        parsed_explicit = parse_targets(base_urls, effective_path)[0]
        host_part, _, port_part = parsed_explicit.address.rpartition(":")
        if host_part and port_part.isdigit():
            advertise_host = host_part.strip("[]")
            inferred_port = int(port_part)

    process = subprocess.Popen(list(command))
    close_reg = auto_register(
        monitor_url,
        port=inferred_port,
        advertise_host=advertise_host,
        metrics_path=effective_path,
        labels=merged_label_map,
        scrape_interval_s=parse_duration_seconds(scrape_interval),
        should_abort=lambda: process.poll() is not None,
    )

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread() and hasattr(
        signal, "SIGTERM"
    ):
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _forward_sigterm(_signum: int, _frame: object) -> None:
            if process.poll() is None:
                process.terminate()

        signal.signal(signal.SIGTERM, _forward_sigterm)

    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                return process.wait(timeout=5)
        return process.returncode or 130
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        close_reg()
