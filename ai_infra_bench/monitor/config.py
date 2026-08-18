import ipaddress
import json
import os
import re
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlsplit

from ai_infra_bench.utils.req import sanitize_url

DURATION_PATTERN = re.compile(r"^[1-9][0-9]*(?:ms|s|m|h|d|w|y)$")


class ConfigurationError(ValueError):
    """Raised when monitoring configuration is invalid."""


@dataclass(frozen=True)
class ScrapeTarget:
    base_url: str
    scheme: str
    address: str
    metrics_path: str
    label: str

    @property
    def metrics_url(self) -> str:
        return f"{self.scheme}://{self.address}{self.metrics_path}"


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    prometheus_config: Path
    prometheus_data: Path
    grafana_config: Path
    grafana_data: Path
    grafana_plugins: Path
    grafana_provisioning: Path
    dashboard_path: Path
    logs: Path


def validate_duration(value: str) -> str:
    if not DURATION_PATTERN.fullmatch(value):
        raise ConfigurationError(
            f"invalid duration {value!r}; use a positive value such as 500ms, 5s, or 24h"
        )
    return value


def _normalize_metrics_path(metrics_path: str) -> str:
    metrics_path = metrics_path.strip()
    if not metrics_path.startswith("/"):
        metrics_path = f"/{metrics_path}"
    parsed = urlsplit(metrics_path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ConfigurationError("--metrics-path must be a URL path without a query")
    return parsed.path.rstrip("/") or "/"


def _format_host_port(host: str, port: int | None) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{rendered_host}:{port}" if port is not None else rendered_host


def parse_targets(
    base_urls: Sequence[str], metrics_path: str = "/metrics"
) -> list[ScrapeTarget]:
    metrics_path = _normalize_metrics_path(metrics_path)
    targets = []
    seen = set()

    for raw_url in base_urls:
        normalized = sanitize_url(raw_url)
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"}:
            raise ConfigurationError(
                f"unsupported scheme in {raw_url!r}; only http and https are supported"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ConfigurationError("credentials are not allowed in --base-urls")
        if not parsed.hostname:
            raise ConfigurationError(f"invalid base URL: {raw_url!r}")
        if parsed.query or parsed.fragment:
            raise ConfigurationError(
                f"base URL {raw_url!r} must not contain a query or fragment"
            )
        if parsed.path not in {"", "/"}:
            raise ConfigurationError(
                f"base URL {raw_url!r} contains an unsupported path; "
                "use --metrics-path for a custom metrics endpoint"
            )
        try:
            address = _format_host_port(parsed.hostname, parsed.port)
        except ValueError as error:
            raise ConfigurationError(f"invalid port in {raw_url!r}: {error}") from error

        key = (parsed.scheme, address, metrics_path)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            ScrapeTarget(
                base_url=f"{parsed.scheme}://{address}",
                scheme=parsed.scheme,
                address=address,
                metrics_path=metrics_path,
                label=f"server-{len(targets)}",
            )
        )

    if not targets:
        raise ConfigurationError("at least one --base-urls value is required")
    return targets


def default_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    return (
        (Path(base).expanduser() if base else Path.home() / ".local" / "state")
        / "ai-infra-bench"
        / "monitoring"
    )


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (
        (Path(base).expanduser() if base else Path.home() / ".cache")
        / "ai-infra-bench"
        / "monitoring"
    )


def create_runtime_layout(
    runtime_dir: Path | None = None, data_dir: Path | None = None
) -> RuntimeLayout:
    root = (runtime_dir or default_state_dir()).expanduser().resolve()
    prometheus_data = (
        data_dir.expanduser().resolve() if data_dir else root / "prometheus-data"
    )
    grafana_root = root / "grafana"
    layout = RuntimeLayout(
        root=root,
        prometheus_config=root / "prometheus.yml",
        prometheus_data=prometheus_data,
        grafana_config=grafana_root / "grafana.ini",
        grafana_data=grafana_root / "data",
        grafana_plugins=grafana_root / "plugins",
        grafana_provisioning=grafana_root / "provisioning",
        dashboard_path=grafana_root / "dashboards" / "sglang-dashboard.json",
        logs=root / "logs",
    )
    for directory in (
        layout.root,
        layout.prometheus_data,
        layout.grafana_data,
        layout.grafana_plugins,
        layout.grafana_provisioning / "datasources",
        layout.grafana_provisioning / "dashboards",
        layout.dashboard_path.parent,
        layout.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def build_prometheus_config(
    targets: Iterable[ScrapeTarget], scrape_interval: str
) -> dict:
    grouped: dict[tuple[str, str], list[ScrapeTarget]] = {}
    for target in targets:
        grouped.setdefault((target.scheme, target.metrics_path), []).append(target)

    scrape_configs = []
    for index, ((scheme, metrics_path), group) in enumerate(grouped.items()):
        job_name = "sglang" if len(grouped) == 1 else f"sglang-{index}"
        scrape_configs.append(
            {
                "job_name": job_name,
                "scheme": scheme,
                "metrics_path": metrics_path,
                "static_configs": [
                    {
                        "targets": [target.address],
                        "labels": {
                            "aib_target": target.label,
                            "aib_base_url": target.base_url,
                        },
                    }
                    for target in group
                ],
            }
        )
    return {
        "global": {
            "scrape_interval": scrape_interval,
            "evaluation_interval": scrape_interval,
        },
        "scrape_configs": scrape_configs,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_prometheus_config(
    path: Path, targets: Iterable[ScrapeTarget], scrape_interval: str
) -> None:
    prometheus_config = build_prometheus_config(targets, scrape_interval)
    _atomic_write_text(path, json.dumps(prometheus_config, indent=2) + "\n")


def _datasource_host(listen_address: str) -> str:
    if listen_address == "0.0.0.0":
        return "127.0.0.1"
    if listen_address in {"::", "[::]"}:
        return "[::1]"
    if listen_address == "localhost":
        return listen_address
    try:
        address = ipaddress.ip_address(listen_address.strip("[]"))
    except ValueError:
        return listen_address
    if address.version == 6:
        return f"[{address}]"
    return str(address)


def build_grafana_ini(
    layout: RuntimeLayout, listen_address: str, grafana_port: int
) -> str:
    return f"""[paths]
data = {layout.grafana_data}
logs = {layout.logs}
plugins = {layout.grafana_plugins}
provisioning = {layout.grafana_provisioning}

[server]
http_addr = {listen_address.strip('[]')}
http_port = {grafana_port}

[analytics]
reporting_enabled = false
check_for_updates = false
check_for_plugin_updates = false

[security]
disable_initial_admin_creation = true

[auth.anonymous]
enabled = true
org_role = Viewer

[auth.basic]
enabled = false

[users]
allow_sign_up = false

[dashboards]
default_home_dashboard_path = {layout.dashboard_path}

[log]
mode = console
"""


def write_runtime_config(
    layout: RuntimeLayout,
    targets: Sequence[ScrapeTarget],
    listen_address: str,
    prometheus_port: int,
    grafana_port: int,
    scrape_interval: str,
) -> None:
    write_prometheus_config(layout.prometheus_config, targets, scrape_interval)

    datasource = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "uid": "aib-prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": (f"http://{_datasource_host(listen_address)}:{prometheus_port}"),
                "isDefault": True,
                "editable": False,
            }
        ],
    }
    datasource_path = layout.grafana_provisioning / "datasources" / "prometheus.yaml"
    datasource_path.write_text(
        json.dumps(datasource, indent=2) + "\n", encoding="utf-8"
    )

    dashboard_provider = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "SGLang",
                "orgId": 1,
                "folder": "SGLang Monitoring",
                "type": "file",
                "disableDeletion": True,
                "updateIntervalSeconds": 10,
                "allowUiUpdates": False,
                "options": {"path": str(layout.dashboard_path.parent)},
            }
        ],
    }
    provider_path = layout.grafana_provisioning / "dashboards" / "dashboard.yaml"
    provider_path.write_text(
        json.dumps(dashboard_provider, indent=2) + "\n", encoding="utf-8"
    )

    dashboard_resource = resources.files("ai_infra_bench.monitor").joinpath(
        "assets", "sglang-dashboard.json"
    )
    layout.dashboard_path.write_text(
        dashboard_resource.read_text(encoding="utf-8"), encoding="utf-8"
    )
    layout.grafana_config.write_text(
        build_grafana_ini(layout, listen_address, grafana_port), encoding="utf-8"
    )
