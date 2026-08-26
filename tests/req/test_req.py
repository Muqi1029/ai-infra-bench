from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_infra_bench.performance.bench_utils import handle_outputs
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.req import main
from ai_infra_bench.utils.req import (
    api_url,
    extract_response_metrics,
    prepare_payload,
    sanitize_url,
)

SPEC_METRICS = {
    "spec_accept_rate": 0.2987012987012987,
    "spec_accept_length": 3.0303030303030303,
    "spec_num_correct_drafts": 69,
    "spec_num_proposed_drafts": 231,
    "spec_verify_ct": 33,
    "spec_correct_drafts_histogram": [4, 10, 7, 7, 3, 1, 0, 1],
}


@pytest.fixture
def stream_handler():
    with patch(
        "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
    ) as handler:
        yield handler


@pytest.fixture
def non_stream_handler():
    response = Mock(status_code=200, text="")
    with (
        patch("ai_infra_bench.req.requests.post", return_value=response) as post,
        patch("ai_infra_bench.req.handle_outputs") as report,
    ):
        yield response, post, report


def test_request_metrics():
    metrics = handle_outputs(
        [OutputMetric(success=True, completion_tokens=20)],
        duration_s=2,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_mode=False,
    )

    values = {row["Metric"]: row["Value"] for row in metrics["Request Result"]}
    assert values["Status"] == "Success"
    assert values["TPS"] == "10.00 tokens/s"


def test_extract_response_metrics_from_choice_meta_info():
    metrics = extract_response_metrics(
        {
            "choices": [{"meta_info": SPEC_METRICS}],
            "usage": {"prompt_tokens_details": {"cached_tokens": 11}},
        }
    )

    expected = SPEC_METRICS | {"cached_tokens": 11}
    assert {key: metrics[key] for key in expected} == expected


def test_extract_response_metrics_reads_standard_reasoning_token_details():
    metrics = extract_response_metrics(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "completion_tokens_details": {"reasoning_tokens": 4},
            }
        }
    )

    assert metrics["reasoning_tokens"] == 4


@pytest.mark.parametrize(
    ("argv", "response_json", "expected"),
    [
        pytest.param(
            ["--disable-stream"],
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "reasoning_content": "reasoning",
                            "content": "answer",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"test"}',
                                    }
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "reasoning_tokens": 1,
                },
            },
            {
                "finish_reason": "stop",
                "content": "answer",
                "reasoning_content": "reasoning",
                "tool_calls": 'Function=lookup\nArgument:{"query":"test"}',
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "reasoning_tokens": 1,
            },
            id="chat",
        ),
        pytest.param(
            ["--disable-stream", "--input-len", "4", "--output-len", "1"],
            {
                "choices": [{"finish_reason": "length", "text": "completion"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
            {
                "finish_reason": "length",
                "content": "completion",
                "prompt_tokens": 4,
                "completion_tokens": 1,
            },
            id="completion",
        ),
    ],
)
def test_non_stream_response(argv, response_json, expected, non_stream_handler):
    response, post, report = non_stream_handler
    response.json.return_value = response_json

    main(argv)

    assert post.call_args.kwargs["json"]["return_meta_info"] is True
    report_args = report.call_args.kwargs
    output = report_args["outputs"][0]
    assert output.success is True
    assert output.latency_ms >= 0
    assert report_args["duration_s"] == pytest.approx(output.latency_ms / 1000)
    assert report_args["max_concurrency"] == 1
    assert report_args["benchmark_mode"] is False
    for field, value in expected.items():
        assert getattr(output, field) == value


def test_non_stream_error(non_stream_handler):
    response, _, report = non_stream_handler
    response.status_code = 500
    response.text = "server error"
    response.raise_for_status.side_effect = RuntimeError("request failed")

    main(["--disable-stream"])

    output = report.call_args.kwargs["outputs"][0]
    assert output.success is False
    assert output.latency_ms >= 0
    assert "Status Code=500" in output.error_message
    assert "server error" in output.error_message


def test_stream_request(stream_handler):
    main([])

    stream_handler.assert_awaited_once()
    url, headers, payload, raw = stream_handler.await_args.args
    assert url == "http://127.0.0.1:30000/v1/chat/completions"
    assert headers == {"Authorization": "Bearer JustKeepMe"}
    assert payload["stream"] is True
    assert "return_meta_info" not in payload
    assert raw is False


def test_length_request(stream_handler):
    argv = ["--input-len", "4", "--output-len", "8", "--seed", "7"]
    main(argv)
    main(argv)

    first, second = (call.args for call in stream_handler.await_args_list)
    url, _, payload, _ = first
    prompt = payload["prompt"]
    assert url == "http://127.0.0.1:30000/v1/completions"
    assert prompt == second[2]["prompt"]
    assert len(prompt) == 4
    assert all(isinstance(token, int) and 0 <= token < 10_000 for token in prompt)
    assert payload["max_tokens"] == 8
    assert payload["ignore_eos"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--input-len", "4"],
        ["--output-len", "8"],
        ["--input-len", "0", "--output-len", "8"],
        ["--input-len", "4", "--output-len", "0"],
        ["--input-len", "4", "--output-len", "8", "--tools"],
        ["--dataset", "gsm8k", "--input-len", "4", "--output-len", "8"],
        ["--dataset", "sharegpt"],
        ["--override-payload", "[]"],
    ],
)
def test_invalid_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_payload_prompt_uses_completions_api(tmp_path, stream_handler):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"prompt": "hello"}', encoding="utf-8")

    main(["--payload-path", str(payload_path)])

    url, _, payload, _ = stream_handler.await_args.args
    assert url == "http://127.0.0.1:30000/v1/completions"
    assert payload["prompt"] == "hello"


def test_dataset_request_is_deterministic(stream_handler):
    requests = [
        {"messages": [{"role": "user", "content": "first"}]},
        {"messages": [{"role": "user", "content": "second"}]},
    ]

    with patch("ai_infra_bench.req.read_packaged_requests", return_value=requests):
        main(["--dataset", "gsm8k", "--seed", "7"])
        main(["--dataset", "gsm8k", "--seed", "7"])

    first, second = (call.args for call in stream_handler.await_args_list)
    assert first[0] == second[0] == "http://127.0.0.1:30000/v1/chat/completions"
    assert first[2]["messages"] == second[2]["messages"]
    assert first[2]["stream"] is True


def test_dataset_rejects_empty_requests():
    with (
        patch("ai_infra_bench.req.read_packaged_requests", return_value=[]),
        pytest.raises(SystemExit),
    ):
        main(["--dataset", "gsm8k"])


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("localhost:8881/v1/", "http://localhost:8881"),
        ("https://example.test/v1", "https://example.test"),
    ],
)
def test_sanitize_url(url, expected):
    assert sanitize_url(url) == expected


def test_api_url_normalizes_base_url():
    assert api_url("localhost:8888/v1", "/v1/models") == (
        "http://localhost:8888/v1/models"
    )


def test_prepare_payload_isolated_and_normalized():
    original = {
        "min_tokens": 0,
        "response_format": {"json_schema": {"schema_": {"type": "object"}}},
    }

    assert prepare_payload(original, "model-name") == {
        "model": "model-name",
        "response_format": {"json_schema": {"schema": {"type": "object"}}},
    }
    assert original["min_tokens"] == 0
    assert "schema_" in original["response_format"]["json_schema"]


def test_prepare_payload_override_and_stream_mode():
    prepared = prepare_payload(
        {"model": "recorded", "stream": False},
        model="cli-model",
        override_payload='{"model": "override-model", "temperature": 0}',
        stream=True,
    )

    assert prepared["model"] == "override-model"
    assert prepared["temperature"] == 0
    assert prepared["stream"] is True
    assert prepared["stream_options"]["include_usage"] is True


def test_prepare_payload_non_stream_mode():
    assert prepare_payload(
        {
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_cached_tokens_details": True,
            "return_spec_tokens_details": True,
        },
        stream=False,
    ) == {"stream": False, "return_meta_info": True}


@pytest.mark.parametrize("override", ["not-json", "[]", '"value"'])
def test_prepare_payload_rejects_invalid_override(override):
    with pytest.raises(ValueError, match="--override-payload"):
        prepare_payload({}, override_payload=override)


def test_extract_response_metrics_prefers_standard_usage_fields():
    metrics = extract_response_metrics(
        {
            "usage": {
                "prompt_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
            "choices": [{"meta_info": {"prompt_tokens": 9, "cached_tokens": 5}}],
            "sglext": {"spec_tokens_details": {"spec_num_proposed_drafts": 4}},
        }
    )

    assert metrics["prompt_tokens"] == 10
    assert metrics["cached_tokens"] == 6
    assert metrics["spec_num_proposed_drafts"] == 4
