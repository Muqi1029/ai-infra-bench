import pytest

from ai_infra_bench.performance.bench import resize_sharegpt_requests
from ai_infra_bench.performance.bench_utils import parse_args, validate_args


class CharacterTokenizer:
    def num_special_tokens_to_add(self):
        return 1

    def encode(self, content):
        return list(content)

    def decode(self, token_ids):
        return "".join(token_ids)


def test_resize_sharegpt_request_repeats_short_prompt():
    original = {"messages": [{"role": "user", "content": "ab"}]}

    requests = resize_sharegpt_requests(
        requests=[original],
        input_len=5,
        output_len=8,
        tokenizer=CharacterTokenizer(),
        num_requests=1,
    )

    assert requests == [
        {
            "messages": [{"role": "user", "content": "abab"}],
            "max_tokens": 8,
            "ignore_eos": True,
        }
    ]
    assert original == {"messages": [{"role": "user", "content": "ab"}]}


def test_resize_sharegpt_request_truncates_long_prompt():
    requests = resize_sharegpt_requests(
        requests=[{"messages": [{"role": "user", "content": "abcdef"}]}],
        input_len=4,
        output_len=2,
        tokenizer=CharacterTokenizer(),
    )

    assert requests[0]["messages"][0]["content"] == "abc"
    assert requests[0]["max_tokens"] == 2


def test_sharegpt_length_arguments_are_validated_together():
    args = parse_args(
        [
            "--dataset",
            "sharegpt",
            "--input-len",
            "128",
            "--output-len",
            "32",
            "--num-requests",
            "4",
            "--model",
            "test-model",
        ]
    )

    validate_args(args)

    assert args.input_len == 128
    assert args.output_len == 32
    assert args.tokenizer is None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            [
                "--dataset",
                "sharegpt",
                "--input-len",
                "128",
                "--output-len",
                "32",
            ],
            "--tokenizer or --model",
        ),
    ],
)
def test_invalid_sharegpt_length_arguments(argv, message):
    with pytest.raises(ValueError, match=message):
        validate_args(parse_args(argv))
