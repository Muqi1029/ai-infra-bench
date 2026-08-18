import asyncio

import pytest

from ai_infra_bench.cli import main as cli_main
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.slo import (
    _send_requests,
    evaluate_conditions,
    load_slo_config,
    parse_slo_config,
    run_slo,
    summarize_outputs,
)

CONFIG = {
    "endpoint": {"base_url": "localhost:30000", "api_key": "test"},
    "request": {"payload": {"messages": [{"role": "user", "content": "hello"}]}},
    "benchmark": {"num_requests": 4, "max_concurrency": 8},
    "search": {"parameter": "max_concurrency", "min": 1, "max": 8},
    "conditions": [
        {"metric": "success_rate", "operator": ">=", "value": 0.99},
        {"metric": "p99_latency_ms", "operator": "<", "value": 100},
    ],
}


def test_load_slo_config_normalizes_endpoint_and_conditions(tmp_path):
    path = tmp_path / "slo.yaml"
    path.write_text(
        """
endpoint:
  base_url: localhost:30000/v1/
request:
  input_len: 4
  output_len: 8
search:
  min: 1
  max: 4
conditions:
  - metric: success_rate
    operator: '>='
    value: 0.9
""",
        encoding="utf-8",
    )

    config = load_slo_config(path)

    assert config.endpoint.base_url == "http://localhost:30000"
    assert config.request.input_len == 4
    assert config.search.parameter == "max_concurrency"
    assert config.conditions[0].value == 0.9


@pytest.mark.parametrize(
    "change",
    [
        {"search": {"parameter": "unsupported", "min": 1, "max": 2}},
        {"conditions": [{"metric": "unknown", "operator": ">", "value": 1}]},
    ],
)
def test_parse_slo_config_rejects_invalid_values(change):
    config = {**CONFIG, **change}
    with pytest.raises(ValueError):
        parse_slo_config(config)


def test_send_requests_calls_shared_request_func(monkeypatch):
    config = parse_slo_config(CONFIG)
    calls = []

    async def fake_request(session, url, payload, sem=None):
        calls.append((session, url, payload, sem))
        return OutputMetric(success=True, latency_ms=10, ttft_ms=2, completion_tokens=2)

    monkeypatch.setattr("ai_infra_bench.slo.request_func", fake_request)
    outputs = asyncio.run(
        _send_requests(
            "session",
            config,
            [{"messages": [{"role": "user", "content": "hi"}]}],
            2,
            float("inf"),
        )
    )

    assert len(outputs) == 1
    assert calls[0][0] == "session"
    assert calls[0][1] == "http://localhost:30000/v1/chat/completions"
    assert "model" not in calls[0][2]
    assert calls[0][3]._value == 2


def test_summarize_outputs_exposes_condition_metrics():
    outputs = [
        OutputMetric(success=True, latency_ms=10, ttft_ms=2, completion_tokens=2),
        OutputMetric(success=False, latency_ms=20),
    ]

    metrics = summarize_outputs(
        outputs, duration_s=1, max_concurrency=2, request_rate=2
    )

    assert metrics["success_rate"] == 0.5
    assert metrics["p99_latency_ms"] == 10
    assert metrics["output_throughput"] == 2


def test_evaluate_conditions_returns_details():
    passed, details = evaluate_conditions(
        {"success_rate": 0.99, "p99_latency_ms": 20},
        parse_slo_config(CONFIG).conditions,
    )

    assert passed is True
    assert all(item["passed"] for item in details)
    assert details[0]["actual"] == 0.99


def test_evaluate_conditions_marks_unavailable_metrics_failed():
    passed, details = evaluate_conditions({}, [parse_slo_config(CONFIG).conditions[1]])

    assert passed is False
    assert details == [
        {
            "metric": "p99_latency_ms",
            "operator": "<",
            "value": 100.0,
            "actual": None,
            "passed": False,
        }
    ]


def test_run_slo_binary_searches_highest_passing_candidate(monkeypatch):
    config = parse_slo_config(CONFIG)

    async def fake_probe(config, candidate):
        return {
            "success_rate": 1.0 if candidate <= 3 else 0.5,
            "p99_latency_ms": candidate * 10,
            "max_concurrency": candidate,
        }

    monkeypatch.setattr("ai_infra_bench.slo._run_probe", fake_probe)
    result = asyncio.run(run_slo(config))

    assert result["status"] == "satisfied"
    assert result["best_value"] == 3
    assert [run["candidate"] for run in result["runs"]] == [4, 2, 3]


def test_cli_dispatches_slo(monkeypatch):
    calls = []
    monkeypatch.setattr("ai_infra_bench.slo.main", lambda argv: calls.append(argv) or 0)

    assert cli_main(["slo", "slo.yaml", "--output", "result.yaml"]) == 0
    assert calls == [["slo.yaml", "--output", "result.yaml"]]
