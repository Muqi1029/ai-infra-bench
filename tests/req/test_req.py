from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_infra_bench.performance.bench_utils import handle_outputs
from ai_infra_bench.performance.struct import OutputMetric
from ai_infra_bench.req import main
from ai_infra_bench.utils.req import SPEC_METRIC_KEYS, extract_response_metrics

SPEC_METRICS = {
    "spec_accept_rate": 0.2987012987012987,
    "spec_accept_length": 3.0303030303030303,
    "spec_num_correct_drafts": 69,
    "spec_num_proposed_drafts": 231,
    "spec_verify_ct": 33,
    "spec_correct_drafts_histogram": [4, 10, 7, 7, 3, 1, 0, 1],
}


def test_request_metrics_include_tps():
    metrics = handle_outputs(
        [OutputMetric(success=True, completion_tokens=20)],
        duration_s=2,
        max_concurrency=1,
        request_rate=float("inf"),
        benchmark_mode=False,
    )

    rows = {row["Metric"]: row for row in metrics["Request Result"]}
    assert rows["TPS"] == {
        "Metric": "TPS",
        "Value": "10.00 tokens/s",
    }


def test_extract_response_metrics_from_choice_meta_info():
    response = {
        "choices": [
            {
                "meta_info": {
                    "prompt_tokens": 12,
                    "completion_tokens": 100,
                    "reasoning_tokens": 103,
                    "cached_tokens": 11,
                    **SPEC_METRICS,
                }
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 100,
            "reasoning_tokens": 103,
            "prompt_tokens_details": {"cached_tokens": 11},
        },
    }

    metrics = extract_response_metrics(response)

    assert {key: metrics[key] for key in SPEC_METRIC_KEYS} == SPEC_METRICS
    assert metrics["cached_tokens"] == 11


def test_http_request_enables_meta_info():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.json.return_value = {
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
    }

    with (
        patch("ai_infra_bench.req.requests.post", return_value=response) as post,
        patch("ai_infra_bench.req.handle_outputs") as handle_outputs,
    ):
        main(["--disable-stream"])

    assert post.call_args.kwargs["json"]["return_meta_info"] is True
    output = handle_outputs.call_args.kwargs["outputs"][0]
    assert output.success is True
    assert output.latency_ms >= 0
    assert output.finish_reason == "stop"
    assert output.content == "answer"
    assert output.reasoning_content == "reasoning"
    assert output.tool_calls == 'Function=lookup\nArgument:{"query":"test"}'
    assert output.prompt_tokens == 4
    assert output.completion_tokens == 2
    assert output.reasoning_tokens == 1
    assert handle_outputs.call_args.kwargs["duration_s"] == pytest.approx(
        output.latency_ms / 1000
    )
    assert handle_outputs.call_args.kwargs["max_concurrency"] == 1
    assert handle_outputs.call_args.kwargs["benchmark_mode"] is False


def test_http_non_stream_completions_response_uses_output_metric():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"finish_reason": "length", "text": "completion"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }

    with (
        patch("ai_infra_bench.req.requests.post", return_value=response),
        patch("ai_infra_bench.req.handle_outputs") as handle_outputs,
    ):
        main(
            [
                "--disable-stream",
                "--input-len",
                "4",
                "--output-len",
                "1",
            ]
        )

    output = handle_outputs.call_args.kwargs["outputs"][0]
    assert output.success is True
    assert output.finish_reason == "length"
    assert output.content == "completion"
    assert output.prompt_tokens == 4
    assert output.completion_tokens == 1


def test_http_non_stream_error_uses_failed_output_metric():
    response = Mock(status_code=500, text="server error")
    response.raise_for_status.side_effect = RuntimeError("request failed")

    with (
        patch("ai_infra_bench.req.requests.post", return_value=response),
        patch("ai_infra_bench.req.handle_outputs") as handle_outputs,
    ):
        main(["--disable-stream"])

    output = handle_outputs.call_args.kwargs["outputs"][0]
    assert output.success is False
    assert output.latency_ms >= 0
    assert "Status Code=500" in output.error_message
    assert "server error" in output.error_message


def test_http_streaming_request_disables_meta_info():
    with patch(
        "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
    ) as handle_stream:
        main([])

    handle_stream.assert_awaited_once()
    url, headers, payload, raw = handle_stream.await_args.args
    assert url == "http://127.0.0.1:30000/v1/chat/completions"
    assert headers == {"Authorization": "Bearer JustKeepMe"}
    assert payload["stream"] is True
    assert "return_meta_info" not in payload
    assert raw is False


def test_length_request_uses_random_input_ids_and_completions_api():
    with patch(
        "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
    ) as handle_stream:
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])

    url, _, payload, _ = handle_stream.await_args.args
    assert url == "http://127.0.0.1:30000/v1/completions"
    assert len(payload["prompt"]) == 4
    assert all(isinstance(token_id, int) for token_id in payload["prompt"])
    assert all(0 <= token_id < 10_000 for token_id in payload["prompt"])
    assert payload["max_tokens"] == 8
    assert payload["ignore_eos"] is True


def test_length_request_is_reproducible():
    with patch(
        "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
    ) as handle_stream:
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])

    assert handle_stream.await_args_list[0].args[2]["prompt"] == (
        handle_stream.await_args_list[1].args[2]["prompt"]
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--input-len", "4"],
        ["--output-len", "8"],
        ["--input-len", "0", "--output-len", "8"],
        ["--input-len", "4", "--output-len", "0"],
        ["--input-len", "4", "--output-len", "8", "--tools"],
        ["--dataset", "gsm8k", "--input-len", "4", "--output-len", "8"],
        ["--override-payload", "[]"],
    ],
)
def test_length_request_validates_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_payload_prompt_uses_completions_api(tmp_path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"prompt": "hello"}', encoding="utf-8")

    with patch(
        "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
    ) as handle_stream:
        main(["--payload-path", str(payload_path)])

    url, _, payload, _ = handle_stream.await_args.args
    assert url == "http://127.0.0.1:30000/v1/completions"
    assert payload["prompt"] == "hello"


@pytest.mark.parametrize("dataset", ["gsm8k", "sharegpt"])
def test_dataset_selects_one_packaged_request_deterministically(dataset):
    dataset_requests = [
        {"messages": [{"role": "user", "content": "first"}]},
        {"messages": [{"role": "user", "content": "second"}]},
    ]

    with (
        patch(
            "ai_infra_bench.req.read_packaged_requests",
            return_value=dataset_requests,
        ),
        patch(
            "ai_infra_bench.req._handle_stream_request", new_callable=AsyncMock
        ) as handle_stream,
    ):
        main(["--dataset", dataset, "--seed", "7"])
        main(["--dataset", dataset, "--seed", "7"])

    first_url, _, first_payload, _ = handle_stream.await_args_list[0].args
    second_url, _, second_payload, _ = handle_stream.await_args_list[1].args
    assert first_url == second_url == "http://127.0.0.1:30000/v1/chat/completions"
    assert first_payload["messages"] == second_payload["messages"]
    assert first_payload["stream"] is True


def test_dataset_rejects_empty_packaged_requests():
    with patch("ai_infra_bench.req.read_packaged_requests", return_value=[]):
        with pytest.raises(SystemExit):
            main(["--dataset", "gsm8k"])
