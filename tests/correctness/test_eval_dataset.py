import asyncio
import json

from ai_infra_bench.correctness.eval_dataset import base
from ai_infra_bench.correctness.eval_dataset.base import Eval


class StubEval(Eval):
    name = "Stub"

    def __init__(self):
        self.results = []

    def _eval(self, response_content, answer, payload=None):
        return answer

    def get_length(self):
        return 1

    def get_payload_and_answer(self, override_payload):
        yield {"messages": []}, True


class FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, content_type=None):
        return {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "reasoning_tokens": 4,
            }
        }

    async def text(self):
        return json.dumps(await self.json())


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, json):
        return FakeResponse()


class UnauthorizedResponse:
    status = 401
    reason = "Unauthorized"
    headers = {"Content-Type": "text/plain"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def text(self):
        return "invalid api key"


class UnauthorizedSession:
    def post(self, url, json):
        return UnauthorizedResponse()


class RecordingProgressBar:
    def __init__(self, total):
        pass

    def set_description(self, description):
        pass

    def update(self, count):
        pass

    def close(self):
        pass


def test_eval_collects_and_sums_token_usage():
    evaluation = StubEval()
    evaluation.eval(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        },
        True,
        {"messages": []},
    )
    evaluation.eval(
        {
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "reasoning_tokens": 3,
            }
        },
        False,
        {"messages": []},
    )

    assert evaluation.results[0].prompt_tokens == 10
    assert evaluation.results[0].completion_tokens == 7
    assert evaluation.results[0].reasoning_tokens == 4
    assert evaluation.token_usage() == {
        "reasoning_tokens": {
            "total": 7,
            "mean": "3.50",
            "p50": "3.50",
            "p90": "3.90",
            "p99": "3.99",
        },
        "prompt_tokens": {
            "total": 22,
            "mean": "11.00",
            "p50": "11.00",
            "p90": "11.80",
            "p99": "11.98",
        },
        "completion_tokens": {
            "total": 12,
            "mean": "6.00",
            "p50": "6.00",
            "p90": "6.80",
            "p99": "6.98",
        },
    }


def test_eval_token_usage_excludes_failed_requests():
    evaluation = StubEval()
    evaluation.eval(
        {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        True,
        {},
    )
    evaluation.add_failed_result(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 100}},
        {},
    )

    assert evaluation.token_usage() == {
        "reasoning_tokens": {
            "total": 0,
            "mean": "0.00",
            "p50": "0.00",
            "p90": "0.00",
            "p99": "0.00",
        },
        "prompt_tokens": {
            "total": 4,
            "mean": "4.00",
            "p50": "4.00",
            "p90": "4.00",
            "p99": "4.00",
        },
        "completion_tokens": {
            "total": 2,
            "mean": "2.00",
            "p50": "2.00",
            "p90": "2.00",
            "p99": "2.00",
        },
    }


def test_eval_token_usage_handles_all_failed_requests():
    evaluation = StubEval()
    evaluation.add_failed_result(None, {})

    assert evaluation.token_usage() == {
        key: {
            "total": 0,
            "mean": "N/A",
            "p50": "N/A",
            "p90": "N/A",
            "p99": "N/A",
        }
        for key in ("reasoning_tokens", "prompt_tokens", "completion_tokens")
    }


def test_eval_runtime_reports_round_token_usage_and_tps(monkeypatch):
    progress_bar = RecordingProgressBar(total=1)
    tables = []
    runtime = object.__new__(base.EvalRuntime)
    runtime.endpoint_url = "http://localhost/v1/chat/completions"
    runtime.max_concurrency = 1
    runtime.api_key = "EMPTY"
    runtime.repeat = 1
    runtime.evals = [StubEval()]
    runtime.override_payload = {}
    runtime.sem = None
    runtime.session = None

    monkeypatch.setattr(base, "tqdm", lambda total: progress_bar)
    monkeypatch.setattr(
        base, "print_table", lambda title, rows: tables.append((title, rows))
    )
    monkeypatch.setattr(
        base, "_create_bench_client_session", lambda *args: FakeSession()
    )
    times = iter((10.0, 12.0))
    monkeypatch.setattr(base, "perf_counter", lambda: next(times))

    asyncio.run(runtime.run())

    assert tables[0] == (
        "Evaluation Summary",
        [
            ["Metric", "Value"],
            ["Evaluation", "Stub"],
            ["Round", "1"],
            ["Total requests", "1"],
            ["Successful requests", "1"],
            ["Failed requests", "0"],
            ["Correct rate", "100.00%"],
            ["Wrong rate", "0.00%"],
            ["Failed rate", "0.00%"],
            ["Max concurrency", "1"],
            ["Duration", "2.00 s"],
            ["TPS", "3.00 tokens/s"],
            ["Total reasoning tokens", "4 tokens"],
            ["Total prompt tokens", "10 tokens"],
            ["Total completion tokens", "6 tokens"],
        ],
    )
    assert tables[1] == (
        "Token Metrics",
        [
            ["Metric", "Mean", "P50", "P90", "P99", "Unit"],
            ["Reasoning tokens", "4.00", "4.00", "4.00", "4.00", "tokens"],
            ["Prompt tokens", "10.00", "10.00", "10.00", "10.00", "tokens"],
            ["Completion tokens", "6.00", "6.00", "6.00", "6.00", "tokens"],
        ],
    )


def test_eval_runtime_logs_authentication_failures(caplog):
    runtime = object.__new__(base.EvalRuntime)
    runtime.endpoint_url = "http://localhost/v1/chat/completions"
    runtime.api_key = "do-not-log-this"
    runtime.sem = asyncio.Semaphore(1)
    runtime.session = UnauthorizedSession()
    evaluation = StubEval()
    progress_bar = RecordingProgressBar(total=1)

    with caplog.at_level("ERROR", logger=base.__name__):
        asyncio.run(runtime._run_one(evaluation, {"messages": []}, True, progress_bar))

    assert evaluation.results[-1].is_failed
    assert "HTTP 401 Unauthorized" in caplog.text
    assert "invalid api key" in caplog.text
    assert "Authentication failed" in caplog.text
    assert "do-not-log-this" not in caplog.text
