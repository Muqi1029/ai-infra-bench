import asyncio
import json
from argparse import Namespace
from dataclasses import asdict

import pytest

from ai_infra_bench.cli import main as cli_main
from ai_infra_bench.performance import bench_utils, session_reply_bench
from ai_infra_bench.performance.bench import (
    compute_random_lens,
    generate_random_requests,
    get_request_url,
    read_requests_with_ts,
    run_requests,
    tool_filter_request,
    validate_args,
)
from ai_infra_bench.performance.bench_utils import handle_outputs
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.utils import device
from ai_infra_bench.utils.draw import format_histogram_percentages


def test_get_first_gpu_info_uses_first_gpu(monkeypatch):
    def run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs["timeout"] == 5
        return Namespace(stdout="GPU 0, 81920\nGPU 1, 40960\n")

    monkeypatch.setattr(device.subprocess, "run", run)

    assert device.get_first_gpu_info() == ("GPU 0", "81920 MiB")


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


def test_random_dataset_arguments_parse_num_requests_as_integer():
    args = bench_utils.parse_args(
        [
            "--dataset",
            "random",
            "--input-len",
            "4",
            "--output-len",
            "8",
            "--num-requests",
            "3",
            "--metric-path",
            "metrics.json",
        ]
    )

    validate_args(args)

    assert args.num_requests == 3
    assert args.metric_path == "metrics.json"


def test_generate_random_requests_uses_requested_lengths():
    bench_utils.set_seed(7)

    requests = generate_random_requests(
        input_len=4,
        output_len=8,
        num_requests=3,
    )

    assert len(requests) == 3
    for payload in requests:
        assert len(payload["prompt"]) == 4
        assert all(isinstance(token_id, int) for token_id in payload["prompt"])
        assert all(0 <= token_id < 10_000 for token_id in payload["prompt"])
        assert payload["max_tokens"] == 8
        assert payload["ignore_eos"] is True


def test_compute_random_lens_samples_sglang_style_range():
    bench_utils.set_seed(7)

    lengths = compute_random_lens(full_len=10, range_ratio=0.5, num=100)

    assert len(lengths) == 100
    assert all(5 <= length <= 10 for length in lengths)
    assert len(set(lengths)) > 1


def test_compute_random_lens_supports_zero_target():
    assert compute_random_lens(full_len=0, range_ratio=0.5, num=3) == [0, 0, 0]


def test_random_dataset_allows_zero_output_length():
    args = bench_utils.parse_args(
        [
            "--dataset",
            "random",
            "--input-len",
            "4",
            "--output-len",
            "0",
            "--num-requests",
            "2",
        ]
    )

    validate_args(args)

    assert all(
        payload["max_tokens"] == 0
        for payload in generate_random_requests(4, 0, num_requests=2)
    )


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_validate_args_rejects_invalid_random_range_ratio(ratio):
    values = {
        "max_concurrency": 1,
        "request_rate": float("inf"),
        "num_warmup_requests": 0,
        "num_requests": None,
        "input_len": 1,
        "output_len": 1,
        "random_range_ratio": ratio,
        "metric_path": None,
        "dump_path": None,
    }

    with pytest.raises(ValueError, match="--random-range-ratio"):
        validate_args(Namespace(**values))


def test_random_dataset_uses_completions_api():
    assert (
        get_request_url("http://localhost:8888", "random")
        == "http://localhost:8888/v1/completions"
    )


def test_read_session_requests_groups_jsonl_files(tmp_path):
    (tmp_path / "b.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"id": 1}\n\n{"id": 3}\n', encoding="utf-8")

    assert session_reply_bench.read_session_requests(str(tmp_path / "*.jsonl")) == [
        [{"id": 1}, {"id": 3}],
        [{"id": 2}],
    ]


def test_session_wrapper_holds_one_semaphore_slot_and_preserves_order(monkeypatch):
    active = 0
    max_active = 0
    completed = {"a": [], "b": []}

    async def fake_request(session, url, payload):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        completed[payload["session"]].append(payload["id"])
        active -= 1
        return OutputMetric(success=True, payload=payload)

    async def no_wait(_request_rate):
        return None

    monkeypatch.setattr(session_reply_bench, "request_func", fake_request)
    monkeypatch.setattr(session_reply_bench, "wait_for_request_interval", no_wait)
    args = Namespace(
        model="session-model",
        override_payload='{"temperature": 0}',
        request_rate=float("inf"),
    )

    async def run():
        semaphore = asyncio.Semaphore(2)
        return await asyncio.gather(
            session_reply_bench.request_func_wrapper(
                args,
                None,
                "url",
                [{"session": "a", "id": 1}, {"session": "a", "id": 2}],
                semaphore,
            ),
            session_reply_bench.request_func_wrapper(
                args,
                None,
                "url",
                [{"session": "b", "id": 3}, {"session": "b", "id": 4}],
                semaphore,
            ),
        )

    outputs = asyncio.run(run())

    assert max_active == 2
    assert completed == {"a": [1, 2], "b": [3, 4]}
    assert outputs[0][0].payload == {
        "session": "a",
        "id": 1,
        "model": "session-model",
        "temperature": 0,
    }


def test_session_parse_args_accepts_positional_and_option_path():
    positional = session_reply_bench.parse_args(["sessions/*.jsonl"])
    option = session_reply_bench.parse_args(
        [
            "--payload-regex-path",
            "sessions/*.jsonl",
            "--metric-path",
            "metrics.jsonl",
        ]
    )

    assert (
        positional.payload_regex_path == option.payload_regex_path == "sessions/*.jsonl"
    )
    assert option.metric_path == "metrics.jsonl"


def test_cli_dispatches_session_bench(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "ai_infra_bench.performance.session_reply_bench.main",
        lambda argv: calls.append(argv) or 0,
    )

    assert cli_main(["session-reply-bench", "sessions/*.jsonl"]) == 0
    assert calls == [["sessions/*.jsonl"]]


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

    async def record_request(session, request_url, payload, sem=None):
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
            pbar=None,
        )
    )

    assert captured == [{"model": "benchmark-model"}]
    assert outputs[0].payload == captured[0]
    assert original == {"min_tokens": 0, "model": "recorded-model"}


def test_run_requests_updates_average_output_throughput(monkeypatch):
    class Progress:
        def __init__(self):
            self.updates = 0
            self.postfixes = []

        def update(self, count):
            self.updates += count

        def set_postfix(self, values):
            self.postfixes.append(values)

    async def return_output(session, request_url, payload, sem=None):
        assert sem is not None
        return OutputMetric(
            success=True,
            completion_tokens=payload["completion_tokens"],
        )

    # Concurrent requests can complete in the same event-loop tick. The running
    # average must use total benchmark time, not the tiny gap between callbacks.
    times = iter([12.0, 12.0001, 20.0])
    progress = Progress()
    monkeypatch.setattr("ai_infra_bench.performance.bench.request_func", return_output)
    monkeypatch.setattr(
        "ai_infra_bench.performance.bench.time.perf_counter", lambda: next(times)
    )

    asyncio.run(
        run_requests(
            session=None,
            request_url="http://localhost/v1/chat/completions",
            requests=[
                {"completion_tokens": 10},
                {"completion_tokens": 20},
                {"completion_tokens": 1},
            ],
            model=None,
            override_payload=None,
            semaphore=asyncio.Semaphore(1),
            pbar=progress,
            benchmark_start_time=10.0,
        )
    )

    assert progress.updates == 3
    assert progress.postfixes == [
        {"Avg TPS": "5.00 tokens/s"},
        {"Avg TPS": "15.00 tokens/s"},
        {"Avg TPS": "3.10 tokens/s"},
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


def test_handle_outputs_supports_single_request_without_usage(monkeypatch):
    tables = []
    monkeypatch.setattr(
        bench_utils,
        "print_table",
        lambda title, rows: tables.append((title, rows)),
    )
    output = OutputMetric(
        success=True,
        latency_ms=12.5,
    )

    handle_outputs(
        [output],
        duration_s=0.0125,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_mode=False,
    )

    result_rows = next(rows for title, rows in tables if title == "Request Result")
    assert ["Status", "Success"] in result_rows
    assert ["Finish reason", "N/A"] in result_rows
    assert not any(title.startswith("Benchmark Summary") for title, _ in tables)
    assert not any(title == "Finish Reason Statistics" for title, _ in tables)

    latency_rows = next(
        rows for title, rows in tables if title == "Latency & Token Metrics"
    )
    assert latency_rows[0] == ["Metric", "Value", "Unit"]
    latency_metrics = {row[0]: row[1:] for row in latency_rows[1:]}
    assert latency_metrics["TTFT"] == ["N/A", "ms"]
    assert latency_metrics["TPOT(ecl the ttft)"] == ["N/A", "ms"]
    assert latency_metrics["Latency"] == ["12.50", "ms"]


def test_handle_outputs_keeps_percentiles_for_multiple_requests(monkeypatch):
    tables = []
    monkeypatch.setattr(
        bench_utils,
        "print_table",
        lambda title, rows: tables.append((title, rows)),
    )
    outputs = [
        OutputMetric(success=True, prompt_tokens=1, latency_ms=10),
        OutputMetric(success=True, prompt_tokens=1, latency_ms=20),
    ]

    handle_outputs(outputs, duration_s=0.02, max_concurrency=2, request_rate=1)

    latency_rows = next(
        rows for title, rows in tables if title == "Latency & Token Metrics"
    )
    assert latency_rows[0] == ["Metric", "Mean", "P50", "P95", "P99", "Unit"]


def test_handle_outputs_keeps_percentiles_for_single_benchmark_request(monkeypatch):
    tables = []
    monkeypatch.setattr(
        bench_utils,
        "print_table",
        lambda title, rows: tables.append((title, rows)),
    )

    handle_outputs(
        [OutputMetric(success=True, latency_ms=10)],
        duration_s=0.01,
        max_concurrency=1,
        request_rate=1,
    )

    latency_rows = next(
        rows for title, rows in tables if title == "Latency & Token Metrics"
    )
    assert latency_rows[0] == ["Metric", "Mean", "P50", "P95", "P99", "Unit"]


def test_handle_outputs_keeps_stable_plot_metadata(monkeypatch):
    tables = []
    monkeypatch.setattr(
        bench_utils,
        "print_table",
        lambda title, rows: tables.append((title, rows)),
    )

    metric_tables = handle_outputs(
        [OutputMetric(success=True, latency_ms=10)],
        duration_s=0.01,
        max_concurrency=8,
        request_rate=1,
        label="server-a",
    )

    assert metric_tables["label"] == "server-a"
    assert metric_tables["max_concurrency"] == 8
    assert metric_tables["timestamp"]
    summary_title = f"Benchmark Summary (server-a {metric_tables['timestamp']})"
    assert any(title == summary_title for title, _ in tables)
    assert summary_title in metric_tables


def test_handle_outputs_dumps_all_outputs_before_filtering(tmp_path):
    outputs = [
        OutputMetric(
            payload={"request_id": 1},
            success=False,
            content="完整回答",
            reasoning_content="reasoning",
            tool_calls='{"name":"tool"}',
            error_message="failed",
        ),
        OutputMetric(
            payload={"request_id": 2},
            success=False,
            error_message="failed again",
        ),
    ]
    dump_path = tmp_path / "all_outputs"
    metrics_path = tmp_path / "failed_metrics.json"

    handle_outputs(
        outputs,
        duration_s=1,
        max_concurrency=1,
        request_rate=1,
        dump_path=str(dump_path),
        metric_path=str(metrics_path),
    )

    dumped_path = tmp_path / "all_outputs.jsonl"
    dumped_text = dumped_path.read_text(encoding="utf-8")
    dumped_outputs = [json.loads(line) for line in dumped_text.splitlines()]
    assert dumped_outputs == [asdict(output) for output in outputs]
    assert "完整回答" in dumped_text
    failed_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert failed_metrics["Benchmark Results"] == [
        {"Metric": "Total requests", "Value": "2"},
        {"Metric": "Successful requests", "Value": "0"},
        {"Metric": "Failed requests", "Value": "2"},
        {"Metric": "Status", "Value": "No successful requests"},
    ]


def test_dump_metric_tables_writes_json_and_appends_jsonl(tmp_path):
    first_metrics = {"Benchmark Summary": [{"Metric": "run", "Value": "1"}]}
    second_metrics = {"Benchmark Summary": [{"Metric": "run", "Value": "2"}]}

    json_path = tmp_path / "metrics.json"
    json_path.write_text('{"stale": true}', encoding="utf-8")
    bench_utils.maybe_dump_metric_tables(first_metrics, str(json_path))
    assert json.loads(json_path.read_text(encoding="utf-8")) == first_metrics

    jsonl_path = tmp_path / "metrics.jsonl"
    bench_utils.maybe_dump_metric_tables(first_metrics, str(jsonl_path))
    bench_utils.maybe_dump_metric_tables(second_metrics, str(jsonl_path))
    assert [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ] == [first_metrics, second_metrics]

    with pytest.raises(ValueError, match=r"\.json or \.jsonl"):
        bench_utils.maybe_dump_metric_tables(
            first_metrics, str(tmp_path / "metrics.txt")
        )


def test_benchmark_summary_includes_total_output_tokens(monkeypatch, tmp_path):
    tables = []
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

    metrics_path = tmp_path / "metrics.json"
    returned_tables = handle_outputs(
        outputs,
        duration_s=2,
        max_concurrency=1,
        request_rate=1,
        metric_path=str(metrics_path),
    )

    assert returned_tables["label"] == ""
    summary_title = f"Benchmark Summary ({returned_tables['timestamp']})"
    summary_rows = next(rows for title, rows in tables if title == summary_title)
    assert ["Total completion tokens", "30 tokens"] in summary_rows
    assert ["Total reasoning tokens", "10 tokens"] in summary_rows

    metric_tables = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary_metrics = {
        row["Metric"]: row["Value"] for row in metric_tables[summary_title]
    }
    assert summary_metrics["Total completion tokens"] == "30 tokens"
    assert summary_metrics["Total reasoning tokens"] == "10 tokens"
    assert "Latency & Token Metrics" in metric_tables
    assert "Finish Reason Statistics" in metric_tables


def test_spec_accept_length_is_weighted_by_verify_ct(monkeypatch):
    tables = []
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
            spec_num_proposed_drafts=10,
            spec_num_correct_drafts=0,
            spec_verify_ct=10,
            spec_accept_length=1.0,
            spec_correct_drafts_histogram=[10],
        ),
        OutputMetric(
            success=True,
            prompt_tokens=2,
            completion_tokens=100,
            spec_num_proposed_drafts=80,
            spec_num_correct_drafts=80,
            spec_verify_ct=20,
            spec_accept_length=5.0,
            spec_correct_drafts_histogram=[0, 0, 0, 0, 20],
        ),
        OutputMetric(
            success=True,
            prompt_tokens=2,
            completion_tokens=50,
        ),
    ]

    handle_outputs(outputs, duration_s=1, max_concurrency=1, request_rate=1)

    spec_rows = next(
        rows for title, rows in tables if title == "Spec Tokens Statistics"
    )
    spec_metrics = {row[0]: row[1] for row in spec_rows[1:]}
    assert spec_metrics["Avg Spec Accept Length(All Verify)"] == "3.67"
    assert "Avg Spec Accept Length" not in spec_metrics
    assert spec_metrics["Avg Spec Accept Rate(All Verify)"] == "88.89%"
    assert "Avg Spec Accept Rate" not in spec_metrics


def test_format_histogram_percentages():
    assert format_histogram_percentages([1, 2, 1]) == "[25.00%, 50.00%, 25.00%]"
    assert format_histogram_percentages([0, 0]) == "[]"
