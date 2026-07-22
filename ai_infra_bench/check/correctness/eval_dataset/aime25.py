import re
from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from ai_infra_bench.check.correctness.eval_dataset.base import Eval
from ai_infra_bench.check.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    resolve_config_path,
)

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def normalize_aime_answer(answer: str) -> Optional[str]:
    """
    Normalize AIME answer to standard format.
    AIME answers are integers from 000 to 999.
    """
    if answer is None:
        return None
    # Remove whitespace and convert to string
    answer = str(answer).strip()
    # Try to extract integer from answer
    try:
        # Handle various formats like "42", "042", "42.0", etc.
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


class AIME25Eval(Eval):
    def __init__(
        self,
        name: str,
        config_path="configs/aime25.yaml",
        dataset_path: str | None = None,
    ):
        self.name = name.replace("_", " ").title()
        self.results = []
        cfg = OmegaConf.load(resolve_config_path(config_path))

        dataset_path = dataset_path or cfg.get("dataset_path", "")
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
        self.default_payload = OmegaConf.to_container(
            cfg.get("payload", {}), resolve=True
        )

    def maybe_truncate(self, num_questions: int | None):
        if num_questions is None:
            return

        if hasattr(self.rows, "select"):
            count = min(num_questions, len(self.rows))
            self.rows = self.rows.select(range(count))
        else:
            self.rows = self.rows[:num_questions]

    def get_length(self) -> int:
        return len(self.rows)

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
        match = re.search(ANSWER_PATTERN, extract_response_text(response_json))
        extracted_answer = match.group(1).strip() if match else None
        return normalize_aime_answer(extracted_answer) == answer


@hydra.main(config_path="configs", config_name="aime25", version_base=None)
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
