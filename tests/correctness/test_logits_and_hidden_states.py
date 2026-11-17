import io

from ai_infra_bench.correctness import hidden_states, logits


def test_logits_extract_compare_and_mae():
    ref = logits.extract_logprobs(
        "http://ref",
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "a",
                                "logprob": -0.1,
                                "top_logprobs": [{"token": "a", "logprob": -0.1}],
                            }
                        ]
                    }
                }
            ]
        },
    )
    tgt = logits.extract_logprobs(
        "http://tgt",
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "a",
                                "logprob": -0.105,
                                "top_logprobs": [{"token": "a", "logprob": -0.105}],
                            }
                        ]
                    }
                }
            ]
        },
    )
    cmp_res = logits.compare_results(ref, tgt, rtol=0.0, atol=0.01)
    assert cmp_res.passed is True
    assert abs(cmp_res.mean_selected_diff - 0.005) < 1e-9


def test_hidden_states_shared_args_and_comparison(monkeypatch):
    monkeypatch.setattr(
        "ai_infra_bench.correctness.common.sys.stdin",
        io.StringIO("stdin prompt"),
    )
    args = hidden_states.parse_args(
        [
            "--base-url",
            "http://127.0.0.1:30000",
            "--model",
            "demo",
            "--prompt",
            "-",
            "--output-len",
            "2",
        ]
    )
    payload = hidden_states.build_payload(args)
    assert payload["model"] == "demo"
    assert payload["max_tokens"] == 2
    assert payload["messages"] == [{"role": "user", "content": "stdin prompt"}]

    ref = hidden_states.extract_hidden_states(
        "http://ref", {"choices": [{"hidden_states": [[1.0, 0.0]]}]}
    )
    tgt = hidden_states.extract_hidden_states(
        "http://tgt", {"choices": [{"hidden_states": [[1.001, 0.0]]}]}
    )
    cmp_res = hidden_states.compare_results(ref, tgt, rtol=0.0, atol=0.01)
    assert cmp_res.passed is True
    assert cmp_res.cosine_similarity > 0.999
