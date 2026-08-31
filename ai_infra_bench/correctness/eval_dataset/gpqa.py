"""Adapter for the GPQA Diamond graduate-level multiple-choice benchmark."""

from __future__ import annotations

import csv
import logging
import os
import random
import re
from typing import Any, Dict, Iterable, Tuple

from omegaconf import OmegaConf

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    resolve_config_path,
    resolve_dataset_path,
)

logger = logging.getLogger(__name__)

GPQA_FIELDS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)
ANSWER_PATTERN = re.compile(r"(?i)(?:final\s+answer|answer)\s*:?\s*([ABCD])\b")
FINAL_OPTION_PATTERN = re.compile(r"(?i)(?<![A-Z])([ABCD])\s*[.)]?\s*$")


def extract_answer(value: Any) -> str | None:
    """Extract a GPQA option letter from a model response."""
    text = str(value or "").strip()
    matches = list(ANSWER_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).upper()
    match = FINAL_OPTION_PATTERN.search(text)
    return match.group(1).upper() if match else None


def format_gpqa_question(question: str, choices: Iterable[str]) -> str:
    options = "\n".join(f"{label}) {choice}" for label, choice in zip("ABCD", choices))
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'Answer: $LETTER' "
        "(without quotes) where LETTER is one of ABCD. Think step by step "
        f"before answering.\n\n{question}\n\n{options}"
    )


def _load_rows(dataset_path: str) -> list[dict[str, Any]]:
    if os.path.isfile(dataset_path):
        with open(dataset_path, "r", encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading GPQA from Hugging Face requires the optional data dependencies. "
            "Install them with `pip install ai-infra-bench[data]`, or pass a local "
            "CSV file."
        ) from exc
    rows = load_dataset(dataset_path, "gpqa_diamond", split="train")
    return [dict(row) for row in rows]


class GPQAEval(Eval):
    """Evaluate GPQA Diamond with a deterministic option permutation."""

    def __init__(
        self,
        name: str,
        config_path="configs/gpqa.yaml",
        dataset_path: str | None = None,
        num_shots: int = 0,
    ):
        self.name = name.replace("_", " ").title()
        self.results = []
        if num_shots:
            logger.info(
                "[%s] GPQA is evaluated zero-shot; ignoring --num-shots=%d",
                self.name,
                num_shots,
            )
        cfg = OmegaConf.load(resolve_config_path(config_path))
        resolved_path = resolve_dataset_path(
            dataset_path or cfg.get("dataset_path", "")
        )
        rows = _load_rows(resolved_path)
        missing = [
            field
            for field in GPQA_FIELDS
            if field not in (rows[0].keys() if rows else ())
        ]
        if missing:
            raise ValueError(
                f"GPQA dataset is missing required columns: {', '.join(missing)}"
            )

        rng = random.Random(0)
        self.rows: list[dict[str, Any]] = []
        for row in rows:
            values = [str(row.get(field) or "").strip() for field in GPQA_FIELDS]
            if not all(values):
                continue
            question, *choices = values
            permutation = rng.sample(range(4), 4)
            shuffled = [choices[index] for index in permutation]
            self.rows.append(
                {
                    "question": question,
                    "A": shuffled[0],
                    "B": shuffled[1],
                    "C": shuffled[2],
                    "D": shuffled[3],
                    "answer": "ABCD"[permutation.index(0)],
                }
            )
        if not self.rows:
            raise ValueError(f"No valid GPQA records found in dataset: {resolved_path}")

        self.prompt_template = cfg.get("prompt_template", "")
        self.default_payload = OmegaConf.to_container(
            cfg.get("payload", {}), resolve=True
        )

    def maybe_truncate(self, num_questions: int | None):
        if num_questions is not None:
            self.rows = self.rows[:num_questions]

    def get_length(self) -> int:
        return len(self.rows)

    def get_payload_and_answer(
        self, override_payload: Dict
    ) -> Iterable[Tuple[Dict, Any]]:
        for row in self.rows:
            payload = generate_payload(
                self.prompt_template,
                row,
                default_payload=self.default_payload,
                override_payload=override_payload,
            )
            yield payload, row["answer"]

    def _eval(self, response_json, answer, payload=None) -> bool:
        return extract_answer(extract_response_text(response_json)) == answer
