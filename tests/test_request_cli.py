"""Exercise shared CLI options through real HTTP requests, without a model server."""

import json
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ai_infra_bench.cli import main


@pytest.fixture
def endpoint():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers.get("Authorization"), payload))
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {
                            "token": "ok",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": "ok", "logprob": -0.1}],
                        }
                    ]
                },
            }
            body = {
                "choices": [choice],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
            if payload.get("stream"):
                choice["delta"] = {"content": "ok"}
                response = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                response = json.dumps(body).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"127.0.0.1:{server.server_port}/v1/", received
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("command", ["req", "bench", "eval-logits"])
@pytest.mark.parametrize("length_option", ["--max-tokens", "--output-len"])
@pytest.mark.parametrize("override_model", [False, True])
def test_shared_request_flags(
    command, length_option, override_model, endpoint, monkeypatch, tmp_path
):
    base_url, received = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    override = {
        "temperature": 0,
        "max_tokens": 3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if override_model:
        override["model"] = "override-model"
    argv = [
        command,
        "--base-url",
        base_url,
        "--model",
        "cli-model",
        "--prompt",
        "你好",
        length_option,
        "8",
        "--override-payload",
        json.dumps(override),
    ]
    if command == "req":
        argv += ["--disable-stream"]
    elif command == "bench":
        argv += [
            "--num-requests",
            "2",
            "--max-concurrency",
            "1",
            "--num-warmup-requests",
            "0",
            "--disable-flush-cache",
            "--metric-path",
            str(tmp_path / "metric.json"),
        ]
    else:
        argv += ["--no-color"]
    assert main(argv) in (None, 0)
    assert len(received) == (2 if command == "bench" else 1)
    for path, authorization, payload in received:
        assert path == "/v1/chat/completions"
        assert authorization == "Bearer environment-key"
        assert payload["model"] == ("override-model" if override_model else "cli-model")
        assert payload["messages"] == [{"role": "user", "content": "你好"}]
        assert payload["max_tokens"] == 3
        assert payload["temperature"] == 0
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_explicit_api_key_overrides_environment(endpoint, monkeypatch):
    base_url, received = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    main(
        ["req", "--base-url", base_url, "--api-key", "explicit-key", "--disable-stream"]
    )
    assert received[0][1] == "Bearer explicit-key"


@pytest.mark.parametrize("command", ["req", "bench", "eval-logits"])
@pytest.mark.parametrize("value", ["not-json", "[]", "null", '"string"'])
def test_invalid_override_is_rejected_before_http(command, value, capsys):
    with pytest.raises(SystemExit) as error:
        main([command, "--base-url", "localhost:1", "--override-payload", value])
    assert error.value.code == 2
    assert "--override-payload" in capsys.readouterr().err


def test_logits_multiple_urls_and_api_key_precedence(endpoint, monkeypatch):
    base_url, received = endpoint
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    assert (
        main(
            [
                "eval-logits",
                "--base-urls",
                base_url,
                base_url,
                "--api-key",
                "explicit-key",
                "--model",
                "my-model",
            ]
        )
        == 0
    )
    assert len(received) == 2
    assert all(
        auth == "Bearer explicit-key" and body["model"] == "my-model"
        for _, auth, body in received
    )


def test_bench_payload_path_selects_completion_endpoint(tmp_path, endpoint):
    base_url, received = endpoint
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"prompt": "hello", "max_tokens": 1}), encoding="utf-8")
    assert (
        main(
            [
                "bench",
                "--base-url",
                base_url,
                "--payload-path",
                str(path),
                "--num-requests",
                "1",
                "--max-concurrency",
                "1",
                "--num-warmup-requests",
                "0",
                "--disable-flush-cache",
                "--metric-path",
                str(tmp_path / "metrics.yaml"),
            ]
        )
        is None
    )
    assert len(received) == 1
    assert received[0][0] == "/v1/completions"
    assert received[0][2]["prompt"] == "hello"


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "--prompt", "hello", "--dataset", "random"],
        ["bench", "--prompt", "hello", "--payload-path", "payload.json"],
        ["req", "--prompt", "hello", "--input-len", "4", "--output-len", "8"],
        ["req", "--prompt", "hello", "--payload-path", "-"],
    ],
)
def test_conflicting_prompt_inputs_are_rejected(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("scenario", "extra_args", "expected_code"),
    [
        ("ok", [], 0),
        ("ok", ["--disable-stream"], 0),
        ("http_500", [], 1),
        ("http_500", ["--disable-stream"], 1),
        ("http_200_html", [], 1),
        ("stream_abort", [], 1),
        ("non_stream_empty_choices", ["--disable-stream"], 1),
    ],
)
def test_req_subprocess_exit_code_matches_request_outcome(
    scenario, extra_args, expected_code
):
    class ScenarioHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if scenario == "http_500":
                status, content_type, body = (
                    500,
                    "application/json",
                    b'{"error":{"message":"server down"}}',
                )
            elif scenario == "http_200_html":
                status, content_type, body = (
                    200,
                    "text/html",
                    b"<html>bad gateway</html>",
                )
            elif scenario == "stream_abort":
                frame = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "partial"},
                                "finish_reason": "abort",
                            }
                        ]
                    }
                )
                status, content_type, body = (
                    200,
                    "text/event-stream",
                    f"data: {frame}\n\ndata: [DONE]\n\n".encode(),
                )
            elif scenario == "non_stream_empty_choices":
                status, content_type, body = 200, "application/json", b'{"choices": []}'
            elif payload.get("stream"):
                frame = json.dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                )
                status, content_type, body = (
                    200,
                    "text/event-stream",
                    f"data: {frame}\n\ndata: [DONE]\n\n".encode(),
                )
            else:
                body_dict = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
                status, content_type, body = (
                    200,
                    "application/json",
                    json.dumps(body_dict).encode(),
                )

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ScenarioHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_infra_bench",
                "req",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}",
                "--prompt",
                "hi",
                *extra_args,
            ],
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    assert result.returncode == expected_code, result.stdout + result.stderr


def test_req_subprocess_connection_refused_and_bad_input_exit_codes(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]

    conn_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_infra_bench",
            "req",
            "--base-url",
            f"http://127.0.0.1:{unused_port}",
            "--prompt",
            "hi",
        ],
        capture_output=True,
        text=True,
    )
    assert conn_result.returncode == 1

    bad_input_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_infra_bench",
            "req",
            "--payload-path",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert bad_input_result.returncode == 2
