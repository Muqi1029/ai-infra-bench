import csv

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.gpqa import GPQAEval, extract_answer
from ai_infra_bench.correctness.eval_dataset.main import parse_args


def _write_csv(path):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question": "Choose?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        )


def test_gpqa_payload_and_scoring(tmp_path):
    source = tmp_path / "gpqa.csv"
    _write_csv(source)
    evaluation = GPQAEval("gpqa", dataset_path=str(source), num_shots=0)
    payload, answer = next(evaluation.get_payload_and_answer({}))

    assert answer in "ABCD"
    assert payload["messages"][-1]["role"] == "user"
    assert evaluation._eval(
        {"choices": [{"message": {"content": f"Answer: {answer}"}}]}, answer
    )
    assert not evaluation._eval(
        {"choices": [{"message": {"content": "Answer: A"}}]},
        "B" if answer != "B" else "C",
    )


def test_gpqa_answer_extraction_and_registration(tmp_path):
    assert extract_answer("Reasoning\nFinal answer: (C)") == "C"
    assert extract_answer("The answer is D") == "D"
    assert extract_answer("B") == "B"
    source = tmp_path / "gpqa.csv"
    _write_csv(source)
    evaluation = Eval.create_from_name("gpqa", dataset_path=str(source), num_shots=0)
    assert isinstance(evaluation, GPQAEval)
    assert parse_args(["--evals", "gpqa", "--num-shots", "0"]).evals == ["gpqa"]
