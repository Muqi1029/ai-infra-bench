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
from ai_infra_bench.monitor.registry_server import (
    RegistryServer,
    _opener_for_url,
    push_metrics_once,
    run_push_agent,
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


def test_monitoring_stack_waits_for_and_stops_real_process(tmp_path, monkeypatch):
    from ai_infra_bench.monitor.process import MonitoringStack

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    stack = MonitoringStack()
    try:
        managed = stack.start(
            "test server",
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(tmp_path),
            ],
            tmp_path / "server.log",
            f"http://127.0.0.1:{port}/index.html",
            timeout=5,
        )
        assert managed.process.poll() is None
    finally:
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


def test_monitor_supports_empty_initial_targets_and_custom_dashboard(tmp_path):
    custom_dash = tmp_path / "my-cluster.json"
    custom_dash.write_text(
        json.dumps({"uid": "custom-1", "title": "My Cluster", "panels": []}),
        encoding="utf-8",
    )
    runtime_dir = tmp_path / "runtime"

    result = main(
        [
            "--runtime-dir",
            str(runtime_dir),
            "--dashboard",
            str(custom_dash),
            "--dry-run",
        ]
    )
    assert result == 0

    sd_targets = json.loads((runtime_dir / "targets.json").read_text(encoding="utf-8"))
    assert sd_targets == []

    prom_cfg = json.loads((runtime_dir / "prometheus.yml").read_text(encoding="utf-8"))
    scrape = prom_cfg["scrape_configs"][0]
    assert scrape["job_name"] == "sglang"
    assert scrape["file_sd_configs"][0]["files"] == [str(runtime_dir / "targets.json")]

    grafana_ini = (runtime_dir / "grafana" / "grafana.ini").read_text(encoding="utf-8")
    assert "org_role = Admin" in grafana_ini
    assert (
        f"default_home_dashboard_path = {runtime_dir / 'grafana' / 'dashboards' / 'my-cluster.json'}"
        in grafana_ini
    )

    provider = json.loads(
        (
            runtime_dir / "grafana" / "provisioning" / "dashboards" / "dashboard.yaml"
        ).read_text(encoding="utf-8")
    )
    assert provider["providers"][0]["allowUiUpdates"] is True
    assert provider["providers"][0]["disableDeletion"] is False

    default_dash_path = runtime_dir / "grafana" / "dashboards" / "sglang-dashboard.json"
    builtin_dash = json.loads(default_dash_path.read_text(encoding="utf-8"))
    var_names = [v["name"] for v in builtin_dash["templating"]["list"]]
    assert {"aib_target", "cluster", "model", "instance"} <= set(var_names)

    # User edits on disk must survive restarts without being overwritten.
    default_dash_path.write_text('{"uid":"edited"}', encoding="utf-8")
    assert main(["--runtime-dir", str(runtime_dir), "--dry-run"]) == 0
    assert json.loads(default_dash_path.read_text(encoding="utf-8")) == {
        "uid": "edited"
    }


def test_registry_server_pull_registration_and_deregistration(tmp_path):
    layout = create_runtime_layout(tmp_path / "runtime")
    write_runtime_config(layout, [], "127.0.0.1", 19090, 13000, "5s")
    registry = TargetRegistry(
        [],
        layout.prometheus_config,
        "5s",
        "/metrics",
        "http://127.0.0.1:19090",
        sd_file_path=layout.targets_sd_path,
    )
    server = RegistryServer(registry, "127.0.0.1", 0)
    server.start()
    registry_url = f"http://127.0.0.1:{server.port}"
    opener = _opener_for_url(registry_url)

    try:
        assert (
            main(
                [
                    "--register",
                    registry_url,
                    "--base-urls",
                    "http://10.12.8.5:30000",
                    "--labels",
                    "cluster=bj-h800",
                    "model=qwen3",
                ]
            )
            == 0
        )

        sd_entries = json.loads(layout.targets_sd_path.read_text(encoding="utf-8"))
        assert len(sd_entries) == 1
        assert sd_entries[0]["targets"] == ["10.12.8.5:30000"]
        assert sd_entries[0]["labels"]["cluster"] == "bj-h800"
        assert sd_entries[0]["labels"]["model"] == "qwen3"
        assert sd_entries[0]["labels"]["aib_target"] == "server-0"
        assert sd_entries[0]["labels"]["__meta_aib_scheme"] == "http"
        assert sd_entries[0]["labels"]["__meta_aib_metrics_path"] == "/metrics"

        with opener.open(f"{registry_url}/api/targets", timeout=3) as resp:
            listed = json.loads(resp.read().decode("utf-8"))
        assert listed["targets"][0]["mode"] == "pull"
        assert listed["targets"][0]["labels"] == {
            "cluster": "bj-h800",
            "model": "qwen3",
        }

        import urllib.request

        dereg_req = urllib.request.Request(
            f"{registry_url}/api/deregister",
            data=b'{"target":"server-0"}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with opener.open(dereg_req, timeout=3) as resp:
            dereg = json.loads(resp.read().decode("utf-8"))
        assert dereg["removed"] == ["server-0"]
        assert json.loads(layout.targets_sd_path.read_text(encoding="utf-8")) == []
    finally:
        server.stop()


def test_registry_server_nat_push_relay_and_ttl_expiry(tmp_path):
    import threading
    import time
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    metrics_body = (
        "# TYPE vllm:num_requests_running gauge\n" "vllm:num_requests_running 7\n"
    )

    class PodMetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            encoded = metrics_body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return None

    pod_server = ThreadingHTTPServer(("127.0.0.1", 0), PodMetricsHandler)
    pod_thread = threading.Thread(target=pod_server.serve_forever, daemon=True)
    pod_thread.start()

    layout = create_runtime_layout(tmp_path / "runtime")
    write_runtime_config(layout, [], "127.0.0.1", 19090, 13000, "5s")
    registry = TargetRegistry(
        [],
        layout.prometheus_config,
        "5s",
        "/metrics",
        "http://127.0.0.1:19090",
        sd_file_path=layout.targets_sd_path,
        push_ttl_s=1.0,
    )
    server = RegistryServer(registry, "127.0.0.1", 0)
    server.start()
    registry_url = f"http://127.0.0.1:{server.port}"
    opener = _opener_for_url(registry_url)

    try:
        # 1) Test raw curl-style text push from a NAT-isolated pod
        raw_push = urllib.request.Request(
            f"{registry_url}/api/push?target=nat-pod-1&cluster=sh-nat&model=deepseek",
            data=metrics_body.encode("utf-8"),
            method="POST",
            headers={"Content-Type": "text/plain"},
        )
        with opener.open(raw_push, timeout=3) as resp:
            assert resp.status == 200

        sd_entries = json.loads(layout.targets_sd_path.read_text(encoding="utf-8"))
        assert len(sd_entries) == 1
        assert sd_entries[0]["targets"] == [f"127.0.0.1:{server.port}"]
        assert (
            sd_entries[0]["labels"]["__meta_aib_metrics_path"]
            == "/pushed/nat-pod-1/metrics"
        )
        assert sd_entries[0]["labels"]["cluster"] == "sh-nat"
        assert sd_entries[0]["labels"]["model"] == "deepseek"

        with opener.open(f"{registry_url}/pushed/nat-pod-1/metrics", timeout=3) as resp:
            assert resp.read().decode("utf-8") == metrics_body

        # Wait for TTL expiry -> auto-prunes stale NAT pod
        time.sleep(1.15)
        with pytest.raises(urllib.error.HTTPError) as http_err:
            opener.open(f"{registry_url}/pushed/nat-pod-1/metrics", timeout=3)
        assert http_err.value.code == 404
        assert json.loads(layout.targets_sd_path.read_text(encoding="utf-8")) == []

        # 2) Test run_push_agent lifecycle (scrape local pod -> push -> auto deregister on stop)
        local_targets = parse_targets(
            [f"http://127.0.0.1:{pod_server.server_port}"],
            labels=(("cluster", "gz-nat"),),
        )
        pushed_id = push_metrics_once(
            registry_url, local_targets[0], target_id="agent-pod"
        )
        assert pushed_id == "agent-pod"
        with opener.open(f"{registry_url}/pushed/agent-pod/metrics", timeout=3) as resp:
            assert "vllm:num_requests_running 7" in resp.read().decode("utf-8")

        stop_event = threading.Event()
        agent_thread = threading.Thread(
            target=run_push_agent,
            args=(registry_url, local_targets, 0.2),
            kwargs={"stop_event": stop_event},
            daemon=True,
        )
        agent_thread.start()
        time.sleep(0.25)
        stop_event.set()
        agent_thread.join(timeout=3)

        # Agent deregistered its pushed target on clean stop; only earlier manual agent-pod remains or expires
        active_labels = {t.label for t in registry.snapshot()}
        assert not any(
            lbl.startswith(f"{local_targets[0].address}") for lbl in active_labels
        )
    finally:
        server.stop()
        pod_server.shutdown()
        pod_server.server_close()
        pod_thread.join(timeout=5)
