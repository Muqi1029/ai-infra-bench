import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_infra_bench.cli import main as cli_main
from ai_infra_bench.monitor.config import create_runtime_layout, write_runtime_config
from ai_infra_bench.monitor.registry_server import RegistryServer, _opener_for_url
from ai_infra_bench.monitor.shell import TargetRegistry
from ai_infra_bench.monitor.telemetry import (
    parse_prometheus_text,
    summarize_telemetry_points,
)


class _MockInferenceBackend:
    """Lightweight OpenAI + SGLang mock server for end-to-end platform tests."""

    def __init__(self, *, logprob_shift: float = 0.0, tp_size: int = 8):
        self.logprob_shift = logprob_shift
        self.tp_size = tp_size
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self.server.server_port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/get_server_info":
                    self._send_json(
                        {
                            "version": "0.4.6",
                            "server_args": {
                                "model_path": "/models/GLM-5-Test",
                                "served_model_name": "glm5-test",
                                "tp_size": outer.tp_size,
                                "kv_cache_dtype": "fp8_e4m3",
                                "enable_metrics": True,
                            },
                        }
                    )
                    return
                if self.path == "/v1/models":
                    self._send_json(
                        {"data": [{"id": "glm5-test", "owned_by": "sglang"}]}
                    )
                    return
                if self.path == "/metrics":
                    text = (
                        "# TYPE sglang:token_usage gauge\n"
                        "sglang:token_usage 0.45\n"
                        "sglang:num_running_reqs 8\n"
                        "sglang:num_queue_reqs 1\n"
                        "sglang:gen_throughput 1250.5\n"
                        "sglang:cache_hit_rate 0.62\n"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(text)))
                    self.end_headers()
                    self.wfile.write(text)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = (
                    json.loads(self.rfile.read(length).decode("utf-8"))
                    if length
                    else {}
                )
                if self.path == "/flush_cache":
                    self._send_json({"status": "flushed"})
                    return

                if body.get("stream"):
                    chunks = [
                        {
                            "choices": [
                                {"delta": {"content": "Hello"}, "finish_reason": None}
                            ]
                        },
                        {
                            "choices": [
                                {
                                    "delta": {"content": " world"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 16,
                                "completion_tokens": 4,
                                "prompt_tokens_details": {"cached_tokens": 4},
                            },
                        },
                    ]
                    sse_bytes = (
                        "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode(
                            "utf-8"
                        )
                        + b"data: [DONE]\n\n"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(sse_bytes)))
                    self.end_headers()
                    self.wfile.write(sse_bytes)
                    return

                lp = -0.05 + outer.logprob_shift
                resp = {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "<think>step by step</think>\n#### 72",
                            },
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "Paris",
                                        "logprob": lp,
                                        "top_logprobs": [
                                            {"token": "Paris", "logprob": lp},
                                            {"token": "Lyon", "logprob": -3.2},
                                        ],
                                    }
                                ]
                            },
                            "hidden_states": [[1.0, 2.0 + outer.logprob_shift, 3.0]],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "reasoning_tokens": 3,
                    },
                }
                self._send_json(resp)

            def log_message(self, *_args):
                return None

        return Handler


def _json_req(opener, url: str, method: str = "GET", payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with opener.open(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_parse_prometheus_text_and_window_summary():
    sample_a = parse_prometheus_text(
        "# HELP sglang:token_usage KV usage\n"
        'sglang:token_usage{model_name="glm5"} 0.35\n'
        "sglang:num_running_reqs 12\n"
        "sglang:num_queue_reqs 0\n"
        "sglang:cache_hit_rate 0.5\n"
        "sglang:generation_tokens_total 1000\n"
    )
    sample_a["ts"] = 100.0
    assert sample_a["kv_cache_pct"] == 35.0
    assert sample_a["running_reqs"] == 12
    assert sample_a["cache_hit_pct"] == 50.0

    sample_b = parse_prometheus_text(
        'sglang:token_usage{model_name="glm5"} 0.72\n'
        "sglang:num_running_reqs 32\n"
        "sglang:num_queue_reqs 4\n"
        "sglang:generation_tokens_total 3000\n"
    )
    sample_b["ts"] = 102.0

    window = summarize_telemetry_points([sample_a, sample_b])
    assert window["sample_count"] == 2
    assert window["summary"]["kv_cache_peak_pct"] == 72.0
    assert window["summary"]["max_waiting_reqs"] == 4
    # Derived from generation_tokens_total delta (2000 tokens / 2.0s = 1000 tok/s)
    assert window["summary"]["server_gen_tps_peak"] == 1000.0


def test_service_session_lifecycle_and_config_diff(tmp_path):
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
    server = RegistryServer(
        registry, "127.0.0.1", 0, prometheus_port=19090, grafana_port=13000
    )
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    opener = _opener_for_url(base)

    try:
        # 1. Studio HTML & platform info served cleanly
        with opener.open(f"{base}/", timeout=3) as resp:
            assert resp.status == 200
            assert b"AI Infra Bench Studio" in resp.read()
        _, info = _json_req(opener, f"{base}/api/platform/info")
        assert info["prometheus_port"] == 19090
        assert info["grafana_port"] == 13000

        # 2. First service startup registers -> online Session 1
        _json_req(
            opener,
            f"{base}/api/register",
            "POST",
            {
                "url": "http://10.12.8.14:30000",
                "labels": {"cluster": "bj", "quant": "bf16"},
                "server_info": {"server_args": {"tp_size": 8, "enable_cp_attn": False}},
            },
        )
        _, s_list = _json_req(opener, f"{base}/api/sessions")
        assert len(s_list["sessions"]) == 1
        sess_1_id = s_list["sessions"][0]["session_id"]
        assert s_list["sessions"][0]["status"] == "online"

        # Repeated registration while online updates same session without duplicating
        _json_req(
            opener,
            f"{base}/api/register",
            "POST",
            {
                "url": "http://10.12.8.14:30000",
                "labels": {"cluster": "bj", "quant": "bf16"},
            },
        )
        _, s_list = _json_req(opener, f"{base}/api/sessions")
        assert len(s_list["sessions"]) == 1

        # 3. Service stops -> deregisters -> Session 1 moves to 'offline', kept in DB
        _json_req(opener, f"{base}/api/deregister", "POST", {"target": "server-0"})
        _, s_list = _json_req(opener, f"{base}/api/sessions")
        assert s_list["sessions"][0]["status"] == "offline"

        # 4. New service starts on SAME host:port with new flags -> creates Session 2!
        _json_req(
            opener,
            f"{base}/api/register",
            "POST",
            {
                "url": "http://10.12.8.14:30000",
                "labels": {"cluster": "bj", "quant": "nvfp4"},
                "server_info": {"server_args": {"tp_size": 8, "enable_cp_attn": True}},
            },
        )
        _, s_list = _json_req(opener, f"{base}/api/sessions")
        assert len(s_list["sessions"]) == 2
        sess_2_id = next(
            s["session_id"] for s in s_list["sessions"] if s["status"] == "online"
        )
        assert sess_2_id != sess_1_id

        # 5. Compare Session 1 vs Session 2 highlights exact flag/label diffs
        _, cmp_doc = _json_req(
            opener, f"{base}/api/compare?sessions={sess_1_id},{sess_2_id}"
        )
        diff_map = {
            d["key"]: (d["baseline"], d["candidate"]) for d in cmp_doc["config_diffs"]
        }
        assert diff_map["enable_cp_attn"] == (False, True)
        assert diff_map["quant"] == ("bf16", "nvfp4")
    finally:
        server.stop()


def test_end_to_end_bench_slo_logits_hidden_and_ingest(tmp_path):
    ref_backend = _MockInferenceBackend(logprob_shift=0.0, tp_size=8)
    cand_backend = _MockInferenceBackend(logprob_shift=0.002, tp_size=4)
    ref_backend.start()
    cand_backend.start()

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
    platform_url = f"http://127.0.0.1:{server.port}"
    opener = _opener_for_url(platform_url)

    try:
        # Register candidate backend
        _json_req(
            opener,
            f"{platform_url}/api/register",
            "POST",
            {"url": cand_backend.base_url, "labels": {"cluster": "test-gpu"}},
        )
        _, live = _json_req(opener, f"{platform_url}/api/monitor/live")
        assert len(live["targets"]) == 1
        assert live["targets"][0]["latest"]["kv_cache_pct"] == 45.0

        # Probe endpoint
        _, probed = _json_req(
            opener,
            f"{platform_url}/api/probe",
            "POST",
            {"base_url": cand_backend.base_url},
        )
        assert probed["server_info"]["engine"] == "sglang"
        assert probed["server_info"]["tp_size"] == 4
        assert probed["metrics_available"] is True

        # Submit a combined Suite job (eval_logits + eval_hidden_states + bench + slo)
        _, job = _json_req(
            opener,
            f"{platform_url}/api/jobs",
            "POST",
            {
                "target": "server-0",
                "session_name": "glm5-cand-suite",
                "job_type": "suite",
                "params": {
                    "steps": [
                        {
                            "type": "eval_logits",
                            "params": {
                                "ref_base_url": ref_backend.base_url,
                                "atol": 0.01,
                                "max_tokens": 1,
                            },
                        },
                        {
                            "type": "eval_hidden_states",
                            "params": {
                                "ref_base_url": ref_backend.base_url,
                                "atol": 0.01,
                            },
                        },
                        {
                            "type": "bench",
                            "params": {
                                "dataset": "random",
                                "input_len": 32,
                                "output_len": 8,
                                "num_requests": 4,
                                "num_warmup_requests": 1,
                                "max_concurrency": [1, 2],
                                "goodput": "ttft:5000 tpot:5000",
                            },
                        },
                        {
                            "type": "slo",
                            "params": {
                                "parameter": "max_concurrency",
                                "min": 1,
                                "max": 2,
                                "input_len": 32,
                                "output_len": 8,
                                "num_requests": 2,
                                "warmup_requests": 0,
                            },
                        },
                    ]
                },
            },
        )

        # Wait for suite job completion
        deadline = time.monotonic() + 15.0
        final_job = job
        while time.monotonic() < deadline:
            _, final_job = _json_req(opener, f"{platform_url}/api/jobs/{job['job_id']}")
            if final_job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(0.15)

        assert final_job["status"] == "succeeded", final_job.get("logs_text")

        # Verify persisted session detail: 3 bench runs (c=1, c=2, slo) + 2 correctness runs
        _, detail = _json_req(
            opener, f"{platform_url}/api/sessions/{final_job['session_id']}"
        )
        assert len(detail["bench_runs"]) == 3
        bench_c1 = detail["bench_runs"][0]
        assert bench_c1["summary"]["successful_requests"] == 4
        assert bench_c1["telemetry"]["summary"]["kv_cache_peak_pct"] == 45.0

        assert len(detail["correctness_runs"]) == 2
        logits_eval = next(
            e for e in detail["correctness_runs"] if e["eval_type"] == "logits"
        )
        assert logits_eval["passed"] is True
        _, cases_doc = _json_req(
            opener, f"{platform_url}/api/evals/{logits_eval['eval_id']}/cases"
        )
        assert len(cases_doc["cases"]) == 1
        assert cases_doc["cases"][0]["ref_token"] == "Paris"

        # Also test CLI --report-to ingestion & `aib studio` alias
        rc = cli_main(
            [
                "eval-logits",
                "--base-urls",
                ref_backend.base_url,
                ref_backend.base_url,
                "--report-to",
                platform_url,
                "--no-color",
            ]
        )
        assert rc == 0
        _, all_sessions = _json_req(opener, f"{platform_url}/api/sessions")
        assert any(
            s["base_url"] == ref_backend.base_url for s in all_sessions["sessions"]
        )
        assert (
            cli_main(["studio", "--runtime-dir", str(tmp_path / "r2"), "--dry-run"])
            == 0
        )
    finally:
        server.stop()
        cand_backend.stop()
        ref_backend.stop()


def test_eval_dataset_job_and_sse_stream_and_session_pin(tmp_path, monkeypatch):
    from ai_infra_bench.correctness.eval_dataset.base import Eval

    class TinyMathEval(Eval):
        name = "tiny_math"

        def __init__(self, name="tiny_math", **_kwargs):
            self.name = name
            self.results = []
            self.rows = [{"q": "36 * 2", "a": 72}, {"q": "wrong case", "a": 99}]

        def get_payload_and_answer(self, override_payload=None):
            for r in self.rows:
                yield {"messages": [{"role": "user", "content": r["q"]}]}, r["a"]

        def _eval(self, response_content, answer, payload=None):
            msg = response_content["choices"][0]["message"]["content"]
            return str(answer) in msg

    monkeypatch.setattr(
        Eval,
        "create_from_name",
        classmethod(lambda cls, name, **kw: TinyMathEval(name)),
    )

    backend = _MockInferenceBackend()
    backend.start()
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
    platform_url = f"http://127.0.0.1:{server.port}"
    opener = _opener_for_url(platform_url)

    sse_events = []

    def _read_sse():
        with opener.open(f"{platform_url}/api/events", timeout=5) as resp:
            for _ in range(6):
                line = resp.readline().decode("utf-8")
                if not line:
                    break
                if line.startswith("event:"):
                    sse_events.append(line.strip())
                    if len(sse_events) >= 2:
                        break

    sse_thread = threading.Thread(target=_read_sse, daemon=True)
    sse_thread.start()

    try:
        _, job = _json_req(
            opener,
            f"{platform_url}/api/jobs",
            "POST",
            {
                "base_url": backend.base_url,
                "session_name": "eval-ds-session",
                "job_type": "eval_dataset",
                "params": {"evals": ["tiny_math"], "max_concurrency": 2, "repeat": 1},
            },
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            _, cur = _json_req(opener, f"{platform_url}/api/jobs/{job['job_id']}")
            if cur["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.1)
        assert cur["status"] == "succeeded", cur.get("logs_text")

        sse_thread.join(timeout=3)
        assert any("job_updated" in e for e in sse_events)

        _, detail = _json_req(
            opener, f"{platform_url}/api/sessions/{job['session_id']}"
        )
        assert len(detail["correctness_runs"]) == 1
        ds_eval = detail["correctness_runs"][0]
        assert ds_eval["correct_rate"] == 0.5

        _, cases_doc = _json_req(
            opener, f"{platform_url}/api/evals/{ds_eval['eval_id']}/cases"
        )
        assert len(cases_doc["cases"]) == 2
        assert cases_doc["cases"][0]["thinking_snippet"] == "step by step"
        assert cases_doc["cases"][0]["answer_text"] == "#### 72"
        assert cases_doc["cases"][0]["is_right"] is True
        assert cases_doc["cases"][1]["is_right"] is False

        # Pin baseline & delete session
        _, patched = _json_req(
            opener,
            f"{platform_url}/api/sessions/{job['session_id']}",
            "PATCH",
            {"is_baseline": True, "name": "pinned-base"},
        )
        assert patched["is_baseline"] is True
        assert patched["name"] == "pinned-base"

        status, _ = _json_req(
            opener, f"{platform_url}/api/sessions/{job['session_id']}", "DELETE"
        )
        assert status == 200
    finally:
        server.stop()
        backend.stop()
