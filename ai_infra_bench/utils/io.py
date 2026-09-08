import json
from typing import Any, Dict, List


def _read_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def _dump_jsonl(data: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
