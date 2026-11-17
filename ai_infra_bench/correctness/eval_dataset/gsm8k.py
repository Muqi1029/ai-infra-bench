import json
import logging
import os
import re
from typing import Any, Dict, Tuple

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    generate_payload_from_content,
    read_jsonl,
    resolve_dataset_path,
)

INVALID = -9999999
logger = logging.getLogger(__name__)

_FINAL_GSM8K_RE = re.compile(
    r"(?is)(?:####|(?:final\s+)?answer\b[*_\s]*(?:is)?[*_\s:]+|\\boxed\s*\{)\s*([-+]?\d+(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_number(raw: str):
    try:
        value = float(raw)
    except ValueError:
        return INVALID
    return int(value) if value.is_integer() else value


def get_answer_value(answer_str):
    cleaned = str(answer_str or "").replace(",", "").replace("$", "")
    marker_matches = list(_FINAL_GSM8K_RE.finditer(cleaned))
    if marker_matches:
        return _parse_number(marker_matches[-1].group(1))
    numbers = _NUMBER_RE.findall(cleaned)
    if not numbers:
        return INVALID
    return _parse_number(numbers[-1])


def get_one_example(row, include_answer: bool) -> str:
    prompt = f"Question: {row['question']}\nAnswer:"
    if include_answer:
        prompt += f" {row['answer']}"
    return prompt


def get_few_shot_examples(rows, num_shots: int) -> str:
    return "".join(
        get_one_example(rows[i], include_answer=True) + "\n\n" for i in range(num_shots)
    )


class GSM8KEval(Eval):

    def __init__(
        self,
        name: str,
        config_path="configs/gsm8k.yaml",
        dataset_path: str | None = None,
        num_shots: int = 5,
    ):
        self.name = name.replace("_", " ").title()
        self.results = []
        self.num_shots = num_shots
        cfg = self.load_config(config_path)

        dataset_path = resolve_dataset_path(dataset_path or cfg.get("dataset_path", ""))
        if not os.path.exists(dataset_path):
            from datasets import load_dataset

            rows = load_dataset(dataset_path, name="main", split="test")
        else:
            rows = list(read_jsonl(dataset_path))

        self.few_shot_prompt = ""
        if self.num_shots:
            if len(rows) <= self.num_shots:
                raise ValueError(
                    f"GSM8K dataset has {len(rows)} examples but num_shots="
                    f"{self.num_shots} requires at least {self.num_shots + 1}."
                )
            self.few_shot_prompt = get_few_shot_examples(rows, self.num_shots)
            if hasattr(rows, "select"):
                self.rows = rows.select(range(self.num_shots, len(rows)))
            else:
                self.rows = rows[self.num_shots :]
        else:
            self.rows = rows

        self.prompt_template = cfg.get("prompt_template", "")
        self.default_payload = dict(cfg.get("payload") or {})
        if self.default_payload:
            logger.info(
                f"[{self.name}] Default Payload: {json.dumps(self.default_payload, indent=2, ensure_ascii=False)}"
            )

    def get_payload_and_answer(self, override_payload) -> Tuple[Dict, Any]:
        for row in self.rows:
            row = dict(row)
            if self.num_shots:
                prompt_content = self.few_shot_prompt + get_one_example(
                    row, include_answer=False
                )
                payload = generate_payload_from_content(
                    prompt_content,
                    default_payload=self.default_payload,
                    override_payload=override_payload,
                )
            else:
                payload = generate_payload(
                    self.prompt_template,
                    row,
                    default_payload=self.default_payload,
                    override_payload=override_payload,
                )
            yield payload, get_answer_value(row["answer"])

    def _eval(self, body, answer, payload=None):
        return get_answer_value(extract_response_text(body)) == answer
