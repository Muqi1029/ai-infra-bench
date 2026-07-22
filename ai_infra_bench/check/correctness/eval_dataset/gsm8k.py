import ast
import os
import re
from typing import Any, Dict, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from ai_infra_bench.check.correctness.eval_dataset.base import Eval
from ai_infra_bench.check.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    read_jsonl,
    resolve_config_path,
)

INVALID = -9999999


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


class GSM8KEval(Eval):

    def __init__(
        self,
        name: str,
        config_path="configs/gsm8k.yaml",
        dataset_path: str | None = None,
    ):
        self.name = name.replace("_", " ").title()
        self.results = []
        cfg = OmegaConf.load(resolve_config_path(config_path))

        dataset_path = dataset_path or cfg.get("dataset_path", "")
        if not os.path.exists(dataset_path):
            from datasets import load_dataset

            self.rows = load_dataset(dataset_path, name="main", split="test")
        else:
            self.rows = list(read_jsonl(dataset_path))

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
            yield payload, get_answer_value(row["answer"])

    def _eval(self, body, answer, payload=None):
        return get_answer_value(extract_response_text(body)) == answer


@hydra.main(config_path="configs", config_name="gsm8k", version_base=None)
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
