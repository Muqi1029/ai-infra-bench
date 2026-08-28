import argparse
import concurrent.futures
import ipaddress
import logging
import os
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import List, Optional, Sequence

from ai_infra_bench.monitor.binaries import BinaryError, resolve_binary
from ai_infra_bench.monitor.config import (
    DEFAULT_METRICS_PATH,
    ConfigurationError,
    ScrapeTarget,
    create_runtime_layout,
    default_cache_dir,
    parse_targets,
    validate_duration,
    write_runtime_config,
)
from ai_infra_bench.monitor.process import (
    MonitoringStack,
    ProcessError,
    ensure_port_available,
    grafana_command,
    prometheus_command,
)
from ai_infra_bench.monitor.shell import MonitorShell, TargetRegistry

logger = logging.getLogger(__name__)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return port


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _duration(value: str) -> str:
    try:
        return validate_duration(value)
    except ConfigurationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _display_host(listen_address: str) -> str:
    host = listen_address.strip("[]")
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if ":" in host else host


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aib monitor",
        description="Run local Prometheus and Grafana services for SGLang metrics",
    )
    parser.add_argument(
        "--base-urls",
        nargs="+",
        required=True,
        metavar="URL",
        help="SGLang server URLs; metrics are read from the URL plus the metrics path",
    )
    parser.add_argument(
        "--runtime",
        choices=["auto", "system", "download"],
        default="auto",
        help="Use system binaries or download verified standalone binaries (default: %(default)s)",
    )
    parser.add_argument("--prometheus-bin", type=Path, help="Prometheus executable")
    parser.add_argument("--grafana-bin", type=Path, help="Grafana executable")
    parser.add_argument(
        "--grafana-home", type=Path, help="Grafana home containing conf/defaults.ini"
    )
    parser.add_argument(
        "--listen-address", default="127.0.0.1", help="Service bind address"
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow a non-loopback bind address",
    )
    parser.add_argument("--prometheus-port", type=_port, default=9090)
    parser.add_argument("--grafana-port", type=_port, default=3000)
    parser.add_argument(
        "--metric-path",
        default=DEFAULT_METRICS_PATH,
        help="Path suffix appended to each target URL (default: %(default)s)",
    )
    parser.add_argument("--scrape-interval", type=_duration, default="5s")
    parser.add_argument("--retention-time", type=_duration, default="24h")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, help="Prometheus TSDB directory")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--startup-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--target-timeout", type=_positive_float, default=2.0)
    parser.add_argument("--skip-target-check", action="store_true")
    parser.add_argument("--no-grafana", action="store_true", help="Run only Prometheus")
    parser.add_argument("--open", action="store_true", help="Open Grafana in a browser")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configuration without resolving or starting binaries",
    )
    return parser.parse_args(argv)


def _probe_target(
    target: ScrapeTarget, timeout: float
) -> tuple[ScrapeTarget, str | None]:
    request = urllib.request.Request(
        target.metrics_url, headers={"User-Agent": "ai-infra-bench-monitoring"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1024 * 1024)
            if not 200 <= response.status < 300:
                return target, f"HTTP {response.status}"
            if b"sglang:" not in body:
                return (
                    target,
                    "response contains no sglang: metrics; enable --enable-metrics",
                )
    except (OSError, urllib.error.URLError) as error:
        return target, str(error)
    return target, None


def probe_targets(targets: Sequence[ScrapeTarget], timeout: float) -> None:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(targets))
    ) as executor:
        checks = [executor.submit(_probe_target, target, timeout) for target in targets]
        for check in concurrent.futures.as_completed(checks):
            target, error = check.result()
            if error:
                logger.warning(
                    "SGLang metrics check failed for %s: %s",
                    target.metrics_url,
                    error,
                )
            else:
                logger.info("SGLang metrics ready: %s", target.metrics_url)


def _print_urls(
    listen_address: str, prometheus_port: int, grafana_port: int | None
) -> tuple[str, str | None]:
    host = _display_host(listen_address)
    prometheus_url = f"http://{host}:{prometheus_port}"
    grafana_url = None
    print(f"Prometheus: {prometheus_url}")
    if grafana_port is not None:
        grafana_url = (
            f"http://{host}:{grafana_port}/d/sglang-dashboard/"
            "sglang-overview?orgId=1&refresh=5s"
        )
        print(f"Grafana:    {grafana_url}")
    return prometheus_url, grafana_url


def run(args: argparse.Namespace) -> int:
    if args.prometheus_port == args.grafana_port and not args.no_grafana:
        raise ConfigurationError("Prometheus and Grafana must use different ports")
    if not _is_loopback(args.listen_address) and not args.allow_remote:
        raise ConfigurationError(
            "a non-loopback --listen-address exposes the monitoring services; "
            "pass --allow-remote to confirm"
        )

    targets: List[ScrapeTarget] = parse_targets(args.base_urls, args.metric_path)
    layout = create_runtime_layout(args.runtime_dir, args.data_dir)
    write_runtime_config(
        layout,
        targets,
        args.listen_address,
        args.prometheus_port,
        args.grafana_port,
        args.scrape_interval,
    )
    print(f"Targets:    {len(targets)}")
    for target in targets:
        print(f"  {target.label}: {target.metrics_url}")
    print(f"Runtime:    {layout.root}")
    if args.dry_run:
        print(f"Config:     {layout.prometheus_config}")
        print("Dry run complete; no services were started.")
        return 0

    ensure_port_available(args.listen_address, args.prometheus_port, "Prometheus")

    if not args.no_grafana:
        ensure_port_available(args.listen_address, args.grafana_port, "Grafana")
    if not args.skip_target_check:
        probe_targets(targets, args.target_timeout)

    prometheus = resolve_binary(
        "prometheus",
        args.runtime,
        args.cache_dir,
        explicit=args.prometheus_bin,
    )
    grafana = None
    if not args.no_grafana:
        grafana = resolve_binary(
            "grafana",
            args.runtime,
            args.cache_dir,
            explicit=args.grafana_bin,
            grafana_home=args.grafana_home,
        )

    host = _display_host(args.listen_address)
    prometheus_health = f"http://{host}:{args.prometheus_port}/-/ready"
    grafana_health = f"http://{host}:{args.grafana_port}/api/health"
    stack = MonitoringStack()
    try:
        stack.start(
            "Prometheus",
            prometheus_command(
                prometheus.executable,
                layout.prometheus_config,
                layout.prometheus_data,
                args.listen_address,
                args.prometheus_port,
                args.retention_time,
            ),
            layout.logs / "prometheus.log",
            prometheus_health,
            args.startup_timeout,
        )
        if grafana is not None:
            assert grafana.home_path is not None
            environment = os.environ.copy()
            environment["GF_PATHS_PROVISIONING"] = str(layout.grafana_provisioning)
            stack.start(
                "Grafana",
                grafana_command(
                    grafana.executable, grafana.home_path, layout.grafana_config
                ),
                layout.logs / "grafana.log",
                grafana_health,
                args.startup_timeout,
                env=environment,
            )

        prometheus_url, grafana_url = _print_urls(
            args.listen_address,
            args.prometheus_port,
            None if args.no_grafana else args.grafana_port,
        )

        # open
        if args.open and grafana_url:
            webbrowser.open(grafana_url)

        # prepare for shell
        registry = TargetRegistry(
            targets=targets,
            config_path=layout.prometheus_config,
            scrape_interval=args.scrape_interval,
            metrics_path=args.metric_path,
            prometheus_url=prometheus_url,
        )
        MonitorShell(registry, process_check=stack.check).cmdloop()
    except KeyboardInterrupt:
        print("\nStopping monitoring services...")
    finally:
        stack.stop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser_args = parse_args(argv)
    try:
        return run(parser_args)
    except (BinaryError, ConfigurationError, ProcessError) as error:
        print(f"aib monitor: error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"aib monitor: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nMonitoring startup cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
