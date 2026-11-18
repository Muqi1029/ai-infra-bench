import ipaddress
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from ai_infra_bench.utils.req import sanitize_url

DEFAULT_METRICS_PATH = "/metrics"
DURATION_PATTERN = re.compile(r"^([1-9][0-9]*)(ms|s|m|h|d|w|y)$")
LABEL_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_DURATION_UNITS_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 7 * 86400.0,
    "y": 365 * 86400.0,
}


class ConfigurationError(ValueError):
    """Raised when monitoring configuration is invalid."""


@dataclass(frozen=True)
class ScrapeTarget:
    base_url: str
    scheme: str
    address: str
    metrics_path: str
    label: str
    labels: tuple[tuple[str, str], ...] = ()
    mode: str = "pull"

    @property
    def metrics_url(self) -> str:
        return f"{self.scheme}://{self.address}{self.metrics_path}"

    @property
    def label_dict(self) -> dict[str, str]:
        return dict(self.labels)


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
    targets_sd_path: Path | None = None


def validate_duration(value: str) -> str:
    if not DURATION_PATTERN.fullmatch(value):
        raise ConfigurationError(
            f"invalid duration {value!r}; use a positive value such as 500ms, 5s, or 24h"
        )
    return value


def parse_duration_seconds(value: str) -> float:
    match = DURATION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ConfigurationError(
            f"invalid duration {value!r}; use a positive value such as 500ms, 5s, or 24h"
        )
    amount, unit = match.groups()
    return int(amount) * _DURATION_UNITS_SECONDS[unit]


def parse_labels(
    raw_labels: Sequence[str] | Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    if not raw_labels:
        return ()

    pairs: dict[str, str] = {}
    if isinstance(raw_labels, Mapping):
        items = [(str(k), str(v)) for k, v in raw_labels.items()]
    else:
        items = []
        for token in raw_labels:
            for part in token.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ConfigurationError(
                        f"invalid label {part!r}; expected KEY=VALUE"
                    )
                key, val = part.split("=", 1)
                items.append((key.strip(), val.strip()))

    for key, value in items:
        if not LABEL_NAME_PATTERN.fullmatch(key):
            raise ConfigurationError(
                f"invalid label name {key!r}; use letters, digits, and underscores"
            )
        if key.startswith("__"):
            raise ConfigurationError(
                f"label name {key!r} cannot start with reserved '__' prefix"
            )
        pairs[key] = value
    return tuple(sorted(pairs.items()))


def _normalize_metrics_path(metrics_path: str) -> str:
    metrics_path = metrics_path.strip()
    if not metrics_path.startswith("/"):
        metrics_path = f"/{metrics_path}"
    parsed = urlsplit(metrics_path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ConfigurationError("--metric-path must be a URL path without a query")
    return parsed.path.rstrip("/") or "/"


def _format_host_port(host: str, port: int | None) -> str:
    if ":" in host and not (host.startswith("[") and host.endswith("]")):
        try:
            ipaddress.ip_address(host)
        except ValueError as error:
            raise ConfigurationError(f"invalid host {host!r}: {error}") from error
        rendered_host = f"[{host}]"
    else:
        rendered_host = host
    return f"{rendered_host}:{port}" if port is not None else rendered_host


def parse_targets(
    base_urls: Sequence[str],
    metrics_path: str = DEFAULT_METRICS_PATH,
    *,
    allow_empty: bool = False,
    labels: tuple[tuple[str, str], ...] = (),
) -> list[ScrapeTarget]:
    metrics_path = _normalize_metrics_path(metrics_path)
    targets = []
    seen = set()

    for raw_url in base_urls:
        raw_url = raw_url.strip()
        if not raw_url:
            continue
        raw_scheme = urlsplit(raw_url).scheme if "://" in raw_url else ""
        if raw_scheme and raw_scheme not in {"http", "https"}:
            raise ConfigurationError(
                f"unsupported scheme in {raw_url!r}; only http and https are supported"
            )
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
        try:
            address = _format_host_port(parsed.hostname, parsed.port)
        except ValueError as error:
            raise ConfigurationError(f"invalid port in {raw_url!r}: {error}") from error

        base_path = parsed.path.rstrip("/")
        if base_path not in {"", "/v1"}:
            raise ConfigurationError(
                f"base URL {raw_url!r} may only include the optional /v1 path"
            )
        key = (parsed.scheme, address, metrics_path)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            ScrapeTarget(
                base_url=f"{parsed.scheme}://{address}{base_path}",
                scheme=parsed.scheme,
                address=address,
                metrics_path=metrics_path,
                label=f"server-{len(targets)}",
                labels=labels,
                mode="pull",
            )
        )

    if not targets and not allow_empty:
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
        targets_sd_path=root / "targets.json",
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


def build_file_sd_targets(targets: Iterable[ScrapeTarget]) -> list[dict]:
    entries = []
    for target in targets:
        labels = {
            **target.label_dict,
            "aib_target": target.label,
            "aib_base_url": target.base_url,
            "__meta_aib_scheme": target.scheme,
            "__meta_aib_metrics_path": target.metrics_path,
        }
        entries.append({"targets": [target.address], "labels": labels})
    return entries


def build_prometheus_config(
    targets: Iterable[ScrapeTarget],
    scrape_interval: str,
    *,
    sd_file_path: Path | None = None,
    default_metrics_path: str = DEFAULT_METRICS_PATH,
) -> dict:
    if sd_file_path is not None:
        normalized_path = _normalize_metrics_path(default_metrics_path)
        return {
            "global": {
                "scrape_interval": scrape_interval,
                "evaluation_interval": scrape_interval,
            },
            "scrape_configs": [
                {
                    "job_name": "sglang",
                    "metrics_path": normalized_path,
                    "honor_labels": True,
                    "file_sd_configs": [
                        {
                            "files": [str(sd_file_path)],
                            "refresh_interval": scrape_interval,
                        }
                    ],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_aib_scheme"],
                            "regex": "(https?)",
                            "target_label": "__scheme__",
                        },
                        {
                            "source_labels": ["__meta_aib_metrics_path"],
                            "regex": "(/.*)",
                            "target_label": "__metrics_path__",
                        },
                    ],
                }
            ],
        }

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
                            **target.label_dict,
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


def write_sd_targets(path: Path, targets: Iterable[ScrapeTarget]) -> None:
    _atomic_write_text(
        path, json.dumps(build_file_sd_targets(targets), indent=2) + "\n"
    )


def write_prometheus_config(
    path: Path,
    targets: Iterable[ScrapeTarget],
    scrape_interval: str,
    *,
    sd_file_path: Path | None = None,
    default_metrics_path: str = DEFAULT_METRICS_PATH,
) -> None:
    prometheus_config = build_prometheus_config(
        targets,
        scrape_interval,
        sd_file_path=sd_file_path,
        default_metrics_path=default_metrics_path,
    )
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
    layout: RuntimeLayout,
    listen_address: str,
    grafana_port: int,
    home_dashboard_path: Path | None = None,
) -> str:
    default_home = home_dashboard_path or layout.dashboard_path
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
org_role = Admin

[auth.basic]
enabled = false

[users]
allow_sign_up = false

[dashboards]
default_home_dashboard_path = {default_home}

[log]
mode = console
"""


def _provision_custom_dashboards(
    dashboard_source: Path, dashboards_dir: Path
) -> Path | None:
    resolved = dashboard_source.expanduser().resolve()
    if not resolved.exists():
        raise ConfigurationError(f"dashboard path does not exist: {dashboard_source}")

    if resolved.is_file():
        if resolved.suffix.lower() != ".json":
            raise ConfigurationError(
                "--dashboard file must be a .json Grafana dashboard"
            )
        try:
            json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigurationError(
                f"invalid JSON in dashboard {dashboard_source}: {error}"
            ) from error
        destination = dashboards_dir / resolved.name
        if resolved != destination:
            shutil.copy2(resolved, destination)
        return destination

    if resolved.is_dir():
        json_files = sorted(resolved.glob("*.json"))
        if not json_files:
            raise ConfigurationError(
                f"--dashboard directory contains no .json files: {dashboard_source}"
            )
        first_dest: Path | None = None
        for item in json_files:
            try:
                json.loads(item.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ConfigurationError(
                    f"invalid JSON in dashboard {item}: {error}"
                ) from error
            dest = dashboards_dir / item.name
            if item.resolve() != dest.resolve():
                shutil.copy2(item, dest)
            if first_dest is None:
                first_dest = dest
        return first_dest

    raise ConfigurationError(f"unsupported dashboard path: {dashboard_source}")


def write_runtime_config(
    layout: RuntimeLayout,
    targets: Sequence[ScrapeTarget],
    listen_address: str,
    prometheus_port: int,
    grafana_port: int,
    scrape_interval: str,
    *,
    metrics_path: str = DEFAULT_METRICS_PATH,
    dashboard: Path | None = None,
) -> None:
    sd_path = layout.targets_sd_path or (layout.root / "targets.json")
    write_sd_targets(sd_path, targets)
    write_prometheus_config(
        layout.prometheus_config,
        targets,
        scrape_interval,
        sd_file_path=sd_path,
        default_metrics_path=metrics_path,
    )

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
                "editable": True,
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
                "name": "AIB Monitoring",
                "orgId": 1,
                "folder": "AI Infra Bench",
                "type": "file",
                "disableDeletion": False,
                "updateIntervalSeconds": 10,
                "allowUiUpdates": True,
                "options": {"path": str(layout.dashboard_path.parent)},
            }
        ],
    }
    provider_path = layout.grafana_provisioning / "dashboards" / "dashboard.yaml"
    provider_path.write_text(
        json.dumps(dashboard_provider, indent=2) + "\n", encoding="utf-8"
    )

    if not layout.dashboard_path.exists():
        dashboard_resource = resources.files("ai_infra_bench.monitor").joinpath(
            "assets", "sglang-dashboard.json"
        )
        layout.dashboard_path.write_text(
            dashboard_resource.read_text(encoding="utf-8"), encoding="utf-8"
        )

    home_dashboard: Path | None = None
    if dashboard is not None:
        home_dashboard = _provision_custom_dashboards(
            dashboard, layout.dashboard_path.parent
        )

    layout.grafana_config.write_text(
        build_grafana_ini(layout, listen_address, grafana_port, home_dashboard),
        encoding="utf-8",
    )
