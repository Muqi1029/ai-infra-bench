from ai_infra_bench.reply import (
    extract_token_ids,
    parse_args,
    prepare_payload,
    select_payloads,
)


def test_prepare_payload_disables_streaming_and_requests_token_ids():
    record = {
        "payload": {
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    }

    payload = prepare_payload(record)

    assert payload["stream"] is False
    assert payload["return_token_ids"] is True
    assert "stream_options" not in payload
    assert record["payload"]["stream"] is True


def test_extract_token_ids():
    input_ids, output_ids = extract_token_ids(
        {"choices": [{"prompt_token_ids": [1, 2], "token_ids": [3, 4]}]}
    )

    assert input_ids == [1, 2]
    assert output_ids == [3, 4]


def test_parse_resume_from():
    args = parse_args(
        [
            "--base-url",
            "localhost:9298",
            "--payload-regex-path",
            "input.jsonl",
            "--resume-from",
            "100",
        ]
    )

    assert args.resume_from == 100


def test_select_payloads_preserves_original_indices():
    selected = select_payloads(list("abcdef"), resume_from=2, num_requests=5)

    assert selected == [(2, "c"), (3, "d"), (4, "e")]
