import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    read_jsonl,
    resolve_dataset_path,
)

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"(?is)\\boxed\s*\{\s*([^}]+?)\s*\}")
ANSWER_PATTERN = re.compile(r"(?is)(?:final\s+)?answer\b[*_\s]*(?:is)?[*_\s:]+([^\n]+)")


def normalize_aime_answer(answer: str) -> Optional[str]:
    """
    Normalize AIME answer to standard format.
    AIME answers are integers from 000 to 999.
    """
    if answer is None:
        return None
    answer = str(answer).strip().strip("$*")
    boxed = _BOXED_RE.search(answer)
    if boxed:
        answer = boxed.group(1).strip()
    try:
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


def extract_aime_answer(text: str) -> Optional[str]:
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return normalize_aime_answer(boxed[-1].group(1))
    matches = list(ANSWER_PATTERN.finditer(text))
    if matches:
        return normalize_aime_answer(matches[-1].group(1))
    return None


class AIME25Eval(Eval):
    def __init__(
        self,
        name: str,
        config_path="configs/aime25.yaml",
        dataset_path: str | None = None,
        num_shots: int = 0,
    ):
        self.name = name.replace("_", " ").title()
        self.results = []
        cfg = self.load_config(config_path)

        dataset_path = resolve_dataset_path(dataset_path or cfg.get("dataset_path", ""))
        if os.path.isdir(dataset_path):
            dataset1 = read_jsonl(os.path.join(dataset_path, "aime2025-I.jsonl"))
            dataset2 = read_jsonl(os.path.join(dataset_path, "aime2025-II.jsonl"))
        else:
            from datasets import load_dataset

            dataset1 = load_dataset(dataset_path, "AIME2025-I", split="test")
            dataset2 = load_dataset(dataset_path, "AIME2025-II", split="test")
        examples1 = [
            {"question": row["question"], "answer": str(row["answer"])}
            for row in dataset1
        ]
        examples2 = [
            {"question": row["question"], "answer": str(row["answer"])}
            for row in dataset2
        ]
        self.rows = examples1 + examples2

        self.prompt_template = cfg.get("prompt_template", "")
        self.default_payload = dict(cfg.get("payload") or {})
        if self.default_payload:
            logger.info(
                f"[{self.name}] Default Payload: {json.dumps(self.default_payload, indent=2, ensure_ascii=False)}"
            )

    def get_payload_and_answer(self, override_payload) -> Tuple[Dict, Any]:
        for row in self.rows:
            payload = generate_payload(
                self.prompt_template,
                row,
                default_payload=self.default_payload,
                override_payload=override_payload,
            )
            yield payload, normalize_aime_answer(row["answer"])

    def _eval(self, response_json, answer, payload=None):
        return extract_aime_answer(extract_response_text(response_json)) == answer
