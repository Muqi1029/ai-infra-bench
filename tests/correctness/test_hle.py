import json

import pytest

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.hle import (
    HLEEval,
    extract_exact_match_answer,
    extract_multiple_choice_answer,
    normalize_text,
)
from ai_infra_bench.correctness.eval_dataset.main import parse_args


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hle_loads_text_rows_and_skips_images_and_invalid_records(tmp_path, caplog):
    source = tmp_path / "hle.jsonl"
    _write_rows(
        source,
        [
            {"question": "2+2?", "answer": "4", "answer_type": "exactMatch"},
            {
                "question": "Which? A or B",
                "answer": "B",
                "answer_type": "multipleChoice",
                "image": None,
            },
            {
                "question": "Image question",
                "answer": "A",
                "answer_type": "multipleChoice",
                "image": "image.png",
            },
            {"question": "missing type", "answer": "x"},
            {"answer": "missing question", "answer_type": "exactMatch"},
        ],
    )

    with caplog.at_level("WARNING"):
        evaluation = HLEEval("hle", dataset_path=str(source), num_shots=0)

    assert evaluation.get_length() == 2
    assert len(evaluation.skipped_rows) == 3
    assert "skipped 3 HLE records" in caplog.text


def test_hle_payload_and_exact_match_scoring(tmp_path):
    source = tmp_path / "hle.jsonl"
    _write_rows(
        source,
        [{"question": "2+2?", "answer": "4", "answer_type": "exactMatch"}],
    )
    evaluation = HLEEval("hle", dataset_path=str(source), num_shots=0)
    payload, answer = next(evaluation.get_payload_and_answer({"model": "test"}))

    assert payload["model"] == "test"
    assert payload["messages"][-1]["role"] == "user"
    assert answer == ("exactMatch", "4")
    assert evaluation._eval(
        {"choices": [{"message": {"content": "Answer: 4"}}]}, answer
    )
    assert not evaluation._eval(
        {"choices": [{"message": {"content": "Answer: 5"}}]}, answer
    )


def test_hle_multiple_choice_extracts_final_option(tmp_path):
    source = tmp_path / "hle.jsonl"
    _write_rows(
        source,
        [
            {
                "question": "Choose one",
                "answer": "The correct answer is C",
                "answer_type": "multipleChoice",
            }
        ],
    )
    evaluation = HLEEval("hle", dataset_path=str(source), num_shots=0)
    _, answer = next(evaluation.get_payload_and_answer({}))

    assert answer == ("multipleChoice", "C")
    assert extract_multiple_choice_answer("Reasoning...\nFinal answer: (C)") == "C"
    assert evaluation._eval(
        {"choices": [{"message": {"content": "I choose C."}}]}, answer
    )


def test_hle_answer_helpers_normalize_markers():
    assert normalize_text("  Hello\nworld ") == "hello world"
    assert extract_exact_match_answer("work\n\nAnswer: 42") == "42"
    assert extract_exact_match_answer("The answer is 42") == "42"
    assert extract_exact_match_answer(r"\boxed{blue}") == "blue"


def test_hle_is_registered_with_eval_cli():
    args = parse_args(["--evals", "hle", "--num-shots", "0"])
    assert args.evals == ["hle"]
    assert args.num_shots == 0


def test_hle_is_created_by_dataset_name(tmp_path):
    source = tmp_path / "hle.jsonl"
    _write_rows(
        source,
        [{"question": "2+2?", "answer": "4", "answer_type": "exactMatch"}],
    )

    evaluation = Eval.create_from_name("hle", dataset_path=str(source), num_shots=0)

    assert isinstance(evaluation, HLEEval)


def test_hle_requires_supported_records(tmp_path):
    source = tmp_path / "hle.jsonl"
    _write_rows(
        source,
        [
            {"question": "image", "answer": "A", "image": "x"},
            {
                "question": "invalid choice",
                "answer": "",
                "answer_type": "multipleChoice",
            },
        ],
    )
    with pytest.raises(ValueError, match="No supported text HLE records"):
        HLEEval("hle", dataset_path=str(source))
