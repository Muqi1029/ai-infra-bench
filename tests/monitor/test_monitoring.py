import io
import json
import socket
import sys
import tarfile

import pytest

from ai_infra_bench.monitor.binaries import (
    ArchiveSpec,
    BinaryError,
    _download,
    _validate_archive,
)
from ai_infra_bench.monitor.cli import _display_host, main
from ai_infra_bench.monitor.config import (
    ConfigurationError,
    _datasource_host,
    build_prometheus_config,
    create_runtime_layout,
    parse_targets,
    write_runtime_config,
)
from ai_infra_bench.monitor.process import (
    ProcessError,
    ensure_port_available,
    prometheus_command,
)
from ai_infra_bench.monitor.shell import (
    MonitorShell,
    ReloadError,
    TargetRegistry,
    reload_prometheus,
)


def test_parse_targets_normalizes_v1_deduplicates_and_preserves_ipv6():
    targets = parse_targets(
        [
            "localhost:30000/v1/",
            "http://localhost:30000",
            "https://[::1]:30001/v1",
        ]
    )

    assert [target.base_url for target in targets] == [
        "http://localhost:30000",
        "https://[::1]:30001",
    ]
    assert targets[1].metrics_url == "https://[::1]:30001/metrics"
    assert [target.label for target in targets] == ["server-0", "server-1"]


@pytest.mark.parametrize(
    "url",
    [
        "http://user:pass@example.test:30000",
        "http://example.test:30000/v1/extra",
        "http://example.test:30000?source=test",
        "ftp://example.test:30000",
    ],
)
def test_parse_targets_rejects_unsafe_or_ambiguous_urls(url):
    with pytest.raises(ConfigurationError):
        parse_targets([url])


def test_prometheus_config_groups_scheme_and_path():
    targets = parse_targets(["http://a.test:30000", "https://b.test:30001"])
    config = build_prometheus_config(targets, "2s")

    assert config["global"] == {
        "scrape_interval": "2s",
        "evaluation_interval": "2s",
    }
    assert [entry["scheme"] for entry in config["scrape_configs"]] == ["http", "https"]
    assert (
        config["scrape_configs"][1]["static_configs"][0]["labels"]["aib_target"]
        == "server-1"
    )


def test_wildcard_listen_addresses_use_matching_loopback_family():
    assert _display_host("0.0.0.0") == "127.0.0.1"
    assert _datasource_host("0.0.0.0") == "127.0.0.1"
    assert _display_host("::") == "[::1]"
    assert _datasource_host("::") == "[::1]"


def test_write_runtime_config_provisions_dashboard(tmp_path):
    layout = create_runtime_layout(tmp_path / "runtime")
    targets = parse_targets(["127.0.0.1:30000"])

    write_runtime_config(layout, targets, "127.0.0.1", 19090, 13000, "5s")

    prometheus_config = json.loads(layout.prometheus_config.read_text())
    assert prometheus_config["scrape_configs"][0]["job_name"] == "sglang"
    datasource = json.loads(
        (layout.grafana_provisioning / "datasources" / "prometheus.yaml").read_text()
    )
    assert datasource["datasources"][0]["url"] == "http://127.0.0.1:19090"
    dashboard = json.loads(layout.dashboard_path.read_text())
    assert dashboard["uid"] == "sglang-dashboard"
    assert dashboard["templating"]["list"][0]["name"] == "aib_target"


def test_monitor_dry_run_does_not_resolve_or_start_services(tmp_path, capsys):
    result = main(
        [
            "--base-urls",
            "localhost:30000",
            "--metric-path",
            "/custom-metrics",
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Dry run complete" in output
    prometheus_config = json.loads(
        (tmp_path / "runtime" / "prometheus.yml").read_text()
    )
    assert prometheus_config["scrape_configs"][0]["metrics_path"] == "/custom-metrics"


def test_ensure_port_available_detects_bound_socket():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        with pytest.raises(ProcessError, match="cannot bind"):
            ensure_port_available("127.0.0.1", port, "Prometheus")


def test_validate_archive_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../../outside")
        info.size = 0
        archive.addfile(info)

    destination = tmp_path / "extract"
    destination.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        with pytest.raises(BinaryError, match="unsafe path"):
            _validate_archive(archive, destination)


def test_download_wraps_network_timeout(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("network stalled")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    spec = ArchiveSpec("service", "1.0", "https://example.test/archive", "0" * 64)

    with pytest.raises(BinaryError, match="failed to download service"):
        _download(spec, tmp_path / "archive.tar.gz", attempts=1)


def test_monitoring_stack_waits_for_and_stops_real_process(tmp_path):
    from ai_infra_bench.monitor.process import MonitoringStack

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    stack = MonitoringStack()
    managed = stack.start(
        "test server",
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
        ],
        tmp_path / "server.log",
        f"http://127.0.0.1:{port}",
        timeout=5,
    )
    assert managed.process.poll() is None

    stack.stop()

    assert managed.process.poll() is not None


def test_prometheus_command_enables_lifecycle_reload(tmp_path):
    command = prometheus_command(
        tmp_path / "prometheus",
        tmp_path / "prometheus.yml",
        tmp_path / "data",
        "127.0.0.1",
        9090,
        "24h",
    )

    assert "--web.enable-lifecycle" in command


def test_target_registry_add_delete_and_stable_labels(tmp_path):
    config_path = tmp_path / "prometheus.yml"
    initial = parse_targets(["localhost:30000"])
    write_runtime_config(
        create_runtime_layout(tmp_path / "runtime"),
        initial,
        "127.0.0.1",
        9090,
        3000,
        "5s",
    )
    config_path = tmp_path / "runtime" / "prometheus.yml"
    reloads = []
    registry = TargetRegistry(
        initial,
        config_path,
        "5s",
        "/metrics",
        "http://127.0.0.1:9090",
        reload_func=reloads.append,
    )

    added = registry.add(["localhost:30001", "localhost:30000"])
    assert [target.label for target in added.added] == ["server-1"]
    assert [target.label for target in added.existing] == ["server-0"]
    assert reloads == ["http://127.0.0.1:9090"]

    deleted = registry.delete(["server-0", "missing"])
    assert [target.label for target in deleted.removed] == ["server-0"]
    assert deleted.missing == ("missing",)
    assert [target.label for target in registry.targets] == ["server-1"]

    added_again = registry.add(["localhost:30002"])
    assert [target.label for target in added_again.added] == ["server-2"]
    config = json.loads(config_path.read_text())
    labels = [
        item["labels"]["aib_target"]
        for job in config["scrape_configs"]
        for item in job["static_configs"]
    ]
    assert labels == ["server-1", "server-2"]


def test_target_registry_rolls_back_rejected_reload(tmp_path):
    layout = create_runtime_layout(tmp_path / "runtime")
    initial = parse_targets(["localhost:30000"])
    write_runtime_config(layout, initial, "127.0.0.1", 9090, 3000, "5s")
    reload_calls = 0

    def reject_first_reload(url):
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 1:
            raise ReloadError("invalid config")

    registry = TargetRegistry(
        initial,
        layout.prometheus_config,
        "5s",
        "/metrics",
        "http://127.0.0.1:9090",
        reload_func=reject_first_reload,
    )

    with pytest.raises(ReloadError, match="invalid config"):
        registry.add(["localhost:30001"])

    assert reload_calls == 2
    assert [target.label for target in registry.targets] == ["server-0"]
    assert registry.next_label == 1
    config = json.loads(layout.prometheus_config.read_text())
    assert len(config["scrape_configs"][0]["static_configs"]) == 1


def test_monitor_shell_add_list_delete_and_quit(tmp_path, capsys):
    layout = create_runtime_layout(tmp_path / "runtime")
    initial = parse_targets(["localhost:30000"])
    write_runtime_config(layout, initial, "127.0.0.1", 9090, 3000, "5s")
    registry = TargetRegistry(
        initial,
        layout.prometheus_config,
        "5s",
        "/metrics",
        "http://127.0.0.1:9090",
        reload_func=lambda url: None,
    )
    shell = MonitorShell(registry)

    shell.onecmd("add localhost:30001")
    shell.onecmd("list")
    shell.onecmd("delete http://localhost:30000")

    output = capsys.readouterr().out
    assert "Added server-1" in output
    assert "server-0: http://localhost:30000/metrics" in output
    assert "Deleted server-0" in output
    assert shell.onecmd("quit") is True


def test_reload_prometheus_uses_post(monkeypatch):
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def open_request(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)

    reload_prometheus("http://127.0.0.1:9090", timeout=7)

    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:9090/-/reload"
    assert request.get_method() == "POST"
    assert timeout == 7


def test_monitor_shell_cmdloop_processes_input_stream(tmp_path, capsys):
    layout = create_runtime_layout(tmp_path / "runtime")
    initial = parse_targets(["localhost:30000"])
    write_runtime_config(layout, initial, "127.0.0.1", 9090, 3000, "5s")
    reloads = []
    checks = []
    registry = TargetRegistry(
        initial,
        layout.prometheus_config,
        "5s",
        "/metrics",
        "http://127.0.0.1:9090",
        reload_func=reloads.append,
    )
    shell = MonitorShell(registry, process_check=lambda: checks.append(True))
    shell.use_rawinput = False
    shell.stdin = io.StringIO("add localhost:30001\nlist\ndelete server-0\nquit\n")

    shell.cmdloop()

    output = capsys.readouterr().out
    assert "Interactive monitor shell" in output
    assert "Added server-1" in output
    assert "Deleted server-0" in output
    assert len(reloads) == 2
    assert len(checks) == 4
    assert [target.label for target in registry.targets] == ["server-1"]
