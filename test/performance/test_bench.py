import asyncio
import json
from argparse import Namespace

import pytest

from ai_infra_bench.performance import bench_utils
from ai_infra_bench.performance.bench import (
    read_requests_with_ts,
    run_requests,
    tool_filter_request,
    validate_args,
)
from ai_infra_bench.performance.bench_utils import handle_outputs
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils.req import format_histogram_percentages


def test_get_first_gpu_info_uses_first_gpu(monkeypatch):
    def run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 5
        return Namespace(stdout="GPU 0, 81920\nGPU 1, 40960\n")

    monkeypatch.setattr(bench_utils.subprocess, "run", run)

    assert bench_utils.get_first_gpu_info() == ("GPU 0", "81920 MiB")


def test_tool_filter_request_supports_openai_tool_shape():
    unconstrained = {
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"strict": False}}],
        "response_format": None,
    }
    strict = {
        **unconstrained,
        "tools": [{"type": "function", "function": {"strict": True}}],
    }

    assert tool_filter_request(unconstrained)
    assert not tool_filter_request(strict)
    assert not tool_filter_request({**unconstrained, "tool_choice": "required"})


def test_read_requests_with_ts_preserves_duplicate_timestamps(tmp_path):
    timestamp = "2026-08-04_12-00-00.000000"
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            [
                [timestamp, json.dumps({"id": 1})],
                [timestamp, json.dumps({"id": 2})],
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(filter_constrained_grammar_requests=False)

    assert read_requests_with_ts(str(path), args) == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_concurrency": 0}, "--max-concurrency"),
        ({"request_rate": 0}, "--request-rate"),
        ({"num_warmup_requests": -1}, "--num-warmup-requests"),
        ({"num_requests": 0}, "--num-requests"),
    ],
)
def test_validate_args(overrides, message):
    values = {
        "max_concurrency": 1,
        "request_rate": float("inf"),
        "num_warmup_requests": 0,
        "num_requests": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_args(Namespace(**values))


def test_get_request_waits_only_between_requests(monkeypatch):
    waits = []

    async def record_wait(request_rate):
        waits.append(request_rate)

    async def collect():
        return [request async for request in bench_utils.get_request([1, 2, 3], 4)]

    monkeypatch.setattr(bench_utils, "wait_for_request_interval", record_wait)

    assert asyncio.run(collect()) == [1, 2, 3]
    assert waits == [4, 4]


def test_run_requests_prepares_copied_payloads(monkeypatch):
    captured = []
    original = {"min_tokens": 0, "model": "recorded-model"}

    async def record_request(session, request_url, payload, semaphore, progress):
        captured.append(payload)
        return OutputMetric(payload=payload)

    monkeypatch.setattr("ai_infra_bench.performance.bench.request_func", record_request)

    outputs = asyncio.run(
        run_requests(
            session=None,
            request_url="http://localhost/v1/chat/completions",
            requests=[original],
            model="benchmark-model",
            override_payload=None,
            semaphore=asyncio.Semaphore(1),
            progress=None,
        )
    )

    assert captured == [{"model": "benchmark-model"}]
    assert outputs[0].payload == captured[0]
    assert original == {"min_tokens": 0, "model": "recorded-model"}


def test_run_requests_updates_live_output_throughput(monkeypatch):
    class Progress:
        def __init__(self):
            self.updates = 0
            self.postfixes = []

        def update(self, count):
            self.updates += count

        def set_postfix(self, values):
            self.postfixes.append(values)

    async def return_output(session, request_url, payload, semaphore, progress):
        assert progress is None
        return OutputMetric(
            success=True,
            completion_tokens=payload["completion_tokens"],
        )

    times = iter([12.0, 14.0])
    progress = Progress()
    monkeypatch.setattr("ai_infra_bench.performance.bench.request_func", return_output)
    monkeypatch.setattr(
        "ai_infra_bench.performance.bench.time.perf_counter", lambda: next(times)
    )

    asyncio.run(
        run_requests(
            session=None,
            request_url="http://localhost/v1/chat/completions",
            requests=[{"completion_tokens": 10}, {"completion_tokens": 20}],
            model=None,
            override_payload=None,
            semaphore=asyncio.Semaphore(1),
            progress=progress,
            benchmark_start_time=10.0,
        )
    )

    assert progress.updates == 2
    assert progress.postfixes == [
        {"TPS": "5.00 tokens/s"},
        {"TPS": "7.50 tokens/s"},
    ]


def test_output_metric_uses_shared_response_metric_extraction():
    output = OutputMetric()

    output.update_response_metrics(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "sglext": {
                "cached_tokens_details": {"device": 3, "host": 1},
                "spec_tokens_details": {
                    "spec_num_proposed_drafts": 5,
                    "spec_correct_drafts_histogram": [1, 2],
                },
            },
        }
    )

    assert output.prompt_tokens == 10
    assert output.completion_tokens == 2
    assert output.cached_tokens == 4
    assert output.cached_tokens_device == 3
    assert output.cached_tokens_host == 1
    assert output.spec_num_proposed_drafts == 5
    assert output.spec_correct_drafts_histogram == [1, 2]


def test_handle_outputs_supports_zero_cached_tokens():
    output = OutputMetric(
        success=True,
        prompt_tokens=2,
        completion_tokens=1,
        latency_ms=10,
    )

    handle_outputs([output], duration_s=1, max_concurrency=1, request_rate=1)


def test_benchmark_summary_includes_total_output_tokens(monkeypatch):
    tables = []
    monkeypatch.setattr(
        bench_utils,
        "get_first_gpu_info",
        lambda: ("Test GPU", "81920 MiB"),
    )
    monkeypatch.setattr(
        bench_utils,
        "print_table",
        lambda title, rows: tables.append((title, rows)),
    )
    outputs = [
        OutputMetric(
            success=True,
            prompt_tokens=2,
            completion_tokens=10,
            reasoning_tokens=3,
        ),
        OutputMetric(
            success=True,
            prompt_tokens=2,
            completion_tokens=20,
            reasoning_tokens=7,
        ),
        OutputMetric(
            success=False,
            prompt_tokens=2,
            completion_tokens=100,
            reasoning_tokens=100,
        ),
    ]

    handle_outputs(outputs, duration_s=2, max_concurrency=1, request_rate=1)

    summary_rows = next(rows for title, rows in tables if title == "Benchmark Summary")
    assert ["Device info", "Test GPU"] in summary_rows
    assert ["Device memory", "81920 MiB"] in summary_rows
    assert ["Total completion tokens", "30 tokens"] in summary_rows
    assert ["Total reasoning tokens", "10 tokens"] in summary_rows


def test_format_histogram_percentages():
    assert format_histogram_percentages([1, 2, 1]) == "[25.00%, 50.00%, 25.00%]"
    assert format_histogram_percentages([0, 0]) == "[]"
