from ai_infra_bench.utils.req import (
    api_url,
    extract_response_metrics,
    prepare_payload,
    sanitize_url,
)


def test_sanitize_url_removes_only_v1_suffix():
    assert sanitize_url("localhost:8881/v1/") == "http://localhost:8881"
    assert sanitize_url("https://example.test/v1") == "https://example.test"


def test_api_url_joins_normalized_base_url():
    assert api_url("localhost:8888/v1", "/v1/models") == (
        "http://localhost:8888/v1/models"
    )


def test_prepare_payload_copies_and_normalizes_input():
    original = {
        "min_tokens": 0,
        "response_format": {"json_schema": {"schema_": {"type": "object"}}},
    }

    prepared = prepare_payload(original, "model-name")

    assert prepared == {
        "model": "model-name",
        "response_format": {"json_schema": {"schema": {"type": "object"}}},
    }
    assert "min_tokens" in original
    assert "schema_" in original["response_format"]["json_schema"]


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
