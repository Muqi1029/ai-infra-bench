from unittest.mock import Mock, patch

import pytest

from ai_infra_bench.req import (
    SPEC_METRIC_KEYS,
    extract_response_metrics,
    main,
    print_metrics,
)

SPEC_METRICS = {
    "spec_accept_rate": 0.2987012987012987,
    "spec_accept_length": 3.0303030303030303,
    "spec_num_correct_drafts": 69,
    "spec_num_proposed_drafts": 231,
    "spec_verify_ct": 33,
    "spec_correct_drafts_histogram": [4, 10, 7, 7, 3, 1, 0, 1],
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
    response.json.return_value = {"usage": {"completion_tokens": 1}}

    with patch("ai_infra_bench.req.requests.post", return_value=response) as post:
        main(["--disable-stream"])

    assert post.call_args.kwargs["json"]["return_meta_info"] is True


def test_http_streaming_request_disables_meta_info():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [b"data: [DONE]"]

    with patch("ai_infra_bench.req.requests.post", return_value=response) as post:
        main([])

    assert "return_meta_info" not in post.call_args.kwargs["json"]


def test_print_metrics_includes_spec_histogram_percentages():
    with patch("ai_infra_bench.req.print_table") as print_table:
        print_metrics(
            0,
            1,
            metrics={"spec_correct_drafts_histogram": [1, 2, 1]},
        )

    spec_rows = print_table.call_args_list[-1].kwargs["rows"]
    assert (
        "spec_correct_drafts_histogram_percentages",
        "[25.00%, 50.00%, 25.00%]",
    ) in spec_rows


def test_length_request_uses_random_input_ids_and_completions_api():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [b"data: [DONE]"]

    with patch("ai_infra_bench.req.requests.post", return_value=response) as post:
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])

    payload = post.call_args.kwargs["json"]
    assert post.call_args.kwargs["url"] == "http://localhost:8888/v1/completions"
    assert len(payload["prompt"]) == 4
    assert all(isinstance(token_id, int) for token_id in payload["prompt"])
    assert all(0 <= token_id < 10_000 for token_id in payload["prompt"])
    assert payload["max_tokens"] == 8
    assert payload["ignore_eos"] is True


def test_length_request_is_reproducible():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [b"data: [DONE]"]

    with patch("ai_infra_bench.req.requests.post", return_value=response) as post:
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])
        main(["--input-len", "4", "--output-len", "8", "--seed", "7"])

    assert post.call_args_list[0].kwargs["json"]["prompt"] == (
        post.call_args_list[1].kwargs["json"]["prompt"]
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--input-len", "4"],
        ["--output-len", "8"],
        ["--input-len", "0", "--output-len", "8"],
        ["--input-len", "4", "--output-len", "0"],
        ["--input-len", "4", "--output-len", "8", "--tools"],
    ],
)
def test_length_request_validates_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_completions_stream_text_sets_first_token_time():
    response = Mock(status_code=200, text="")
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [
        b'data: {"choices":[{"text":"hello"}]}',
        b"data: [DONE]",
    ]

    with (
        patch("ai_infra_bench.req.requests.post", return_value=response),
        patch("ai_infra_bench.req.print_metrics") as print_metrics,
    ):
        main(["--input-len", "4", "--output-len", "8"])

    assert print_metrics.call_args.args[2] is not None
