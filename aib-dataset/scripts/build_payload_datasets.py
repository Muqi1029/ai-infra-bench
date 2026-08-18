#!/usr/bin/env python3
"""Build OpenAI-compatible payload JSONL files for benchmark datasets."""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "ai_infra_bench_dataset"
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_GSM8K_SOURCE = DEFAULT_DATA_DIR / "gsm8k" / "test.jsonl"
DEFAULT_SHAREGPT_CACHE = (
    Path.home() / ".cache" / "ai-infra-bench-dataset" / SHAREGPT_FILENAME
)
MAX_PAYLOAD_SHARD_BYTES = 4 * 1024 * 1024
PRIVATE_KEY_MARKER = re.compile(r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----")


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Yield values from a top-level JSON array without loading it all at once."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as source:
        buffer = ""
        position = 0
        array_started = False

        while True:
            while position >= len(buffer):
                chunk = source.read(chunk_size)
                if not chunk:
                    if array_started:
                        raise ValueError(f"Unterminated JSON array in {path}")
                    raise ValueError(f"Expected a JSON array in {path}")
                buffer = chunk
                position = 0

            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if position >= len(buffer):
                continue

            if not array_started:
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                array_started = True
                position += 1
                continue

            if buffer[position] == ",":
                position += 1
                continue
            if buffer[position] == "]":
                return

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                chunk = source.read(chunk_size)
                if not chunk:
                    raise ValueError(
                        f"Invalid JSON array in {path}: {error}"
                    ) from error
                buffer = buffer[position:] + chunk
                position = 0
                continue

            yield value
            position = end

            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def download_sharegpt(destination: Path) -> Path:
    if destination.is_file():
        return destination

    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    url = f"{endpoint}/datasets/{SHAREGPT_REPO_ID}/resolve/main/{SHAREGPT_FILENAME}"
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} to {destination}", file=sys.stderr)
    try:
        with (
            urllib.request.urlopen(url) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_payload(output, payload: dict[str, Any]) -> None:
    output.write(serialize_payload(payload))


def serialize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def contains_private_key(content: str) -> bool:
    return PRIVATE_KEY_MARKER.search(content) is not None


def build_gsm8k_payloads(source_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        source_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Invalid GSM8K question at {source_path}:{line_number}"
                )
            write_payload(
                output,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Question: {question}\nAnswer:",
                        }
                    ]
                },
            )
            count += 1
    return count


def first_human_message(record: dict[str, Any]) -> str | None:
    conversations = record.get("conversations", record.get("conversation", []))
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", turn.get("role"))
        if role not in {"human", "user"}:
            continue
        content = turn.get("value", turn.get("content"))
        if isinstance(content, str) and content.strip():
            return content
    return None


def build_sharegpt_payloads(
    source_path: Path,
    output_path: Path,
    limit: int | None = None,
    max_shard_bytes: int = MAX_PAYLOAD_SHARD_BYTES,
) -> tuple[int, int]:
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be >= 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_pattern = f"{output_path.stem}-*.jsonl"
    for stale_path in (output_path, *output_path.parent.glob(shard_pattern)):
        stale_path.unlink(missing_ok=True)

    written = 0
    skipped = 0
    shard_index = 0
    output = None
    output_size = 0
    try:
        for record in iter_json_array(source_path):
            if limit is not None and written >= limit:
                break
            content = first_human_message(record) if isinstance(record, dict) else None
            if content is None or contains_private_key(content):
                skipped += 1
                continue
            payload = {"messages": [{"role": "user", "content": content}]}
            line = serialize_payload(payload)
            line_size = len(line.encode("utf-8"))
            if line_size > max_shard_bytes:
                skipped += 1
                continue
            if output is None or output_size + line_size > max_shard_bytes:
                if output is not None:
                    output.close()
                shard_path = output_path.with_name(
                    f"{output_path.stem}-{shard_index:05d}{output_path.suffix}"
                )
                output = shard_path.open("w", encoding="utf-8")
                output_size = 0
                shard_index += 1
            output.write(line)
            output_size += line_size
            written += 1
    finally:
        if output is not None:
            output.close()
    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "gsm8k", "sharegpt"),
        default="all",
    )
    parser.add_argument(
        "--gsm8k-source",
        type=Path,
        default=DEFAULT_GSM8K_SOURCE,
    )
    parser.add_argument(
        "--sharegpt-source",
        type=Path,
        help="Use a local ShareGPT JSON file instead of downloading it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--sharegpt-limit",
        type=int,
        help="Only convert this many valid ShareGPT records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sharegpt_limit is not None and args.sharegpt_limit < 1:
        raise ValueError("--sharegpt-limit must be >= 1")

    if args.dataset in {"all", "gsm8k"}:
        output_path = args.output_dir / "gsm8k" / "payload.jsonl"
        count = build_gsm8k_payloads(args.gsm8k_source, output_path)
        print(f"Wrote {count} GSM8K payloads to {output_path}")

    if args.dataset in {"all", "sharegpt"}:
        source_path = args.sharegpt_source or download_sharegpt(DEFAULT_SHAREGPT_CACHE)
        output_path = args.output_dir / "sharegpt" / "payload.jsonl"
        count, skipped = build_sharegpt_payloads(
            source_path,
            output_path,
            limit=args.sharegpt_limit,
        )
        print(
            f"Wrote {count} ShareGPT payloads to {output_path}; "
            f"skipped {skipped} invalid records"
        )


if __name__ == "__main__":
    main()
