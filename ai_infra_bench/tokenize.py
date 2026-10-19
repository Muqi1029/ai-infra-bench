"""Encode or decode tokens with a Hugging Face tokenizer."""

import argparse
import json
import sys
from typing import Sequence


def load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_id)
    except ImportError as error:
        raise RuntimeError(
            "Token commands require transformers; install ai-infra-bench[data]"
        ) from error


def _parse_ids(raw: str):
    raw = raw.strip()
    if not raw:
        raise ValueError("token ids must not be empty")
    if raw.startswith(("{", "[")):
        data = json.loads(raw)
        while isinstance(data, dict):
            if "input_ids" not in data:
                raise ValueError("JSON object must contain input_ids")
            data = data["input_ids"]
        return data
    return [int(x) for x in raw.replace(",", " ").split()]


def _is_1d(ids) -> bool:
    return not ids or type(ids[0]) is int


def _decode(tokenizer, ids):
    return (
        tokenizer.decode(ids) if _is_1d(ids) else [_decode(tokenizer, x) for x in ids]
    )


def _count(ids) -> int:
    return len(ids) if _is_1d(ids) else sum(_count(x) for x in ids)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="aib token")
    parser.add_argument(
        "--tokenizer",
        "--model",
        required=True,
        help="Hugging Face tokenizer id or local path",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode token ids instead of encoding text",
    )
    parser.add_argument(
        "--from-stdin", action="store_true", help="Read text or token ids from stdin"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to encode, or token ids / JSON when --decode is set",
    )
    args = parser.parse_args(argv)

    if (args.text is None) == (not args.from_stdin):
        parser.error("provide text or use --from-stdin")

    source = sys.stdin.read().removesuffix("\n") if args.from_stdin else args.text

    try:
        tokenizer = load_tokenizer(args.tokenizer)
        if args.decode:
            ids = _parse_ids(source)
            payload = {
                "text": _decode(tokenizer, ids),
                "ids": ids,
                "count": _count(ids),
            }
        else:
            ids = tokenizer.encode(source)
            payload = {"text": source, "ids": ids, "count": len(ids)}
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
