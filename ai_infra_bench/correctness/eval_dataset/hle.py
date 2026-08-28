"""Text-only adapter for the Humanity's Last Exam dataset.

The published HLE dataset contains both text and image questions. This
adapter intentionally sends only text questions to the chat-completions API;
image questions belong to the separate multimodal/tool benchmark.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Tuple

from omegaconf import OmegaConf

from ai_infra_bench.correctness.eval_dataset.base import Eval
from ai_infra_bench.correctness.eval_dataset.utils import (
    extract_response_text,
    generate_payload,
    read_jsonl,
    resolve_config_path,
    resolve_dataset_path,
)

logger = logging.getLogger(__name__)

_ANSWER_TYPES = {
    "exactmatch": "exactMatch",
    "exact": "exactMatch",
    "multiplechoice": "multipleChoice",
    "mcq": "multipleChoice",
}
_ANSWER_MARKER_RE = re.compile(
    r"(?is)(?:final\s+answer|answer)\b\s*(?:is\s*)?[:\-]?\s*"
    r"(?:\\boxed\s*\{\s*)?([A-Z])(?:\s*\})?\b"
)
_BOXED_RE = re.compile(r"(?is)\\boxed\s*\{\s*(.*?)\s*\}")
_OPTION_RE = re.compile(
    r"(?<![A-Za-z])(?:\(|\[)?([A-Z])(?:\)|\])?[\s.!?]*$", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Normalize answer text without changing its substantive content."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return _WHITESPACE_RE.sub(" ", value).strip()


def _canonical_answer_type(value: Any) -> str | None:
    if value is None:
        return None
    key = re.sub(r"[^a-z]", "", str(value).casefold())
    return _ANSWER_TYPES.get(key)


def extract_multiple_choice_answer(value: Any) -> str | None:
    """Extract the final option letter from an HLE answer or model response."""
    text = str(value or "")
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        boxed_text = boxed[-1].group(1).strip()
        if re.fullmatch(r"[A-Za-z]", boxed_text):
            return boxed_text.upper()

    marked = list(_ANSWER_MARKER_RE.finditer(text))
    if marked:
        return marked[-1].group(1).upper()

    # A final standalone token handles terse responses such as "C" and
    # responses ending in "Therefore, (C)" without matching letters in words.
    option = _OPTION_RE.search(text)
    return option.group(1).upper() if option else None


def extract_exact_match_answer(value: Any) -> str:
    """Extract a final-answer marker when present, otherwise use full output."""
    text = str(value or "").strip()
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()

    markers = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer)\b\s*(?:is\s*)?[:\-]?\s*(.+?)(?=$|\n)",
            text,
        )
    )
    if markers:
        return markers[-1].group(1).strip()
    return text


def _has_image(row: Mapping[str, Any]) -> bool:
    for key in ("image", "images"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (bytes, bytearray)) and not value:
            continue
        if isinstance(value, (list, tuple, dict, set)) and not value:
            continue
        return True
    return False


def _load_local_rows(path: str) -> list[dict[str, Any]]:
    if os.path.isfile(path):
        return list(read_jsonl(path))
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".jsonl")
        )
        if not files:
            raise ValueError(f"HLE dataset directory contains no JSONL files: {path}")
        rows: list[dict[str, Any]] = []
        for filename in files:
            rows.extend(read_jsonl(filename))
        return rows
    return []


def _load_rows(dataset_path: str) -> Iterable[Mapping[str, Any]]:
    local_rows = _load_local_rows(dataset_path)
    if local_rows:
        return local_rows
    if os.path.exists(dataset_path):
        return local_rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading HLE from Hugging Face requires the optional data dependencies. "
            "Install them with `pip install ai-infra-bench[data]`, or pass a local JSONL file."
        ) from exc
    return load_dataset(dataset_path, split="test")


class HLEEval(Eval):
    """Evaluate text-only HLE records using a direct chat-completions call."""

    def __init__(
        self,
        name: str,
        config_path="configs/hle.yaml",
        dataset_path: str | None = None,
        num_shots: int = 0,
    ):
        cfg = OmegaConf.load(resolve_config_path(config_path))
        self.name = cfg.get("name", name.replace("_", " ").title())
        self.results = []
        if num_shots:
            logger.info(
                "[%s] HLE is evaluated zero-shot; ignoring --num-shots=%d",
                self.name,
                num_shots,
            )

        resolved_path = resolve_dataset_path(
            dataset_path or cfg.get("dataset_path", "")
        )
        raw_rows = _load_rows(resolved_path)
        self.rows: list[dict[str, Any]] = []
        self.skipped_rows: list[tuple[int, str]] = []
        for index, raw_row in enumerate(raw_rows, start=1):
            if not isinstance(raw_row, Mapping):
                self.skipped_rows.append((index, "record is not an object"))
                continue
            if _has_image(raw_row):
                self.skipped_rows.append(
                    (index, "image question is not supported by the text adapter")
                )
                continue
            question = raw_row.get("question")
            if not isinstance(question, str) or not question.strip():
                self.skipped_rows.append((index, "missing question"))
                continue
            if "answer" not in raw_row or raw_row.get("answer") is None:
                self.skipped_rows.append((index, "missing answer"))
                continue
            answer_type = _canonical_answer_type(raw_row.get("answer_type"))
            if answer_type is None:
                self.skipped_rows.append((index, "unsupported or missing answer_type"))
                continue
            answer = str(raw_row["answer"])
            expected_answer = (
                extract_multiple_choice_answer(answer)
                if answer_type == "multipleChoice"
                else normalize_text(answer)
            )
            if not expected_answer:
                self.skipped_rows.append((index, "empty or invalid answer"))
                continue
            self.rows.append(
                {
                    "question": question,
                    "answer_type": answer_type,
                    "expected_answer": expected_answer,
                }
            )

        if self.skipped_rows:
            logger.warning(
                "[%s] skipped %d HLE records (%d image, %d invalid)",
                self.name,
                len(self.skipped_rows),
                sum("image question" in reason for _, reason in self.skipped_rows),
                sum("image question" not in reason for _, reason in self.skipped_rows),
            )
        if not self.rows:
            raise ValueError(
                f"No supported text HLE records found in dataset: {resolved_path}"
            )

        self.prompt_template = cfg.get("prompt_template", "")
        self.default_payload = OmegaConf.to_container(
            cfg.get("payload", {}), resolve=True
        )
        if self.default_payload:
            logger.info(
                "[%s] Default Payload: %s",
                self.name,
                json.dumps(self.default_payload, indent=2, ensure_ascii=False),
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
            yield payload, (row["answer_type"], row["expected_answer"])

    def _eval(self, response_json, answer, payload=None) -> bool:
        answer_type, expected = answer
        response = extract_response_text(response_json)
        if answer_type == "multipleChoice":
            return extract_multiple_choice_answer(response) == expected
        return normalize_text(extract_exact_match_answer(response)) == expected
