"""Stdlib-only JSON, JSONL, glob, and packaged dataset I/O helpers."""

import json
import logging
import sys
from datetime import datetime
from glob import glob
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S.%f"


def _read_json(file_path: str) -> Any:
    if file_path == "-":
        return json.load(sys.stdin)
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(file_path: str) -> List[Any]:
    if file_path == "-":
        return [json.loads(line) for line in sys.stdin if line.strip()]
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def _dump_jsonl(data: List[Dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_payload_files(path_pattern: str) -> List[str]:
    candidate = Path(path_pattern)
    if candidate.is_file():
        return [path_pattern]
    return sorted(glob(path_pattern, recursive=True))


def read_requests(payload_path: str) -> List[Dict]:
    if payload_path == "-":
        raw = sys.stdin.read().strip()
        if not raw:
            return []
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        if isinstance(loaded, Mapping):
            return [dict(loaded)]
        if isinstance(loaded, list):
            return [dict(item) for item in loaded]
        raise ValueError("stdin payload must be a JSON object, JSON array, or JSONL")

    requests: List[Dict] = []
    for file_path in resolve_payload_files(payload_path):
        logger.info(f"Reading {file_path}")
        if file_path.endswith(".json"):
            loaded = _read_json(file_path)
            if isinstance(loaded, Mapping):
                requests.append(dict(loaded))
            elif isinstance(loaded, list):
                requests.extend(loaded)
            else:
                raise ValueError(f"{file_path} must contain a JSON object or array")
        elif file_path.endswith(".jsonl"):
            requests.extend(_read_jsonl(file_path))
        else:
            logger.error(f"{file_path} cannot be read. Only support json/jsonl suffix")
    return requests


def read_requests_with_ts(payload_path: str, args: Any = None) -> List[Dict]:
    timestamped_requests = []
    for file_path in resolve_payload_files(payload_path):
        logger.info(f"Reading {file_path}")
        timestamped_requests.extend(_read_json(file_path))

    timestamped_requests.sort(key=lambda item: datetime.strptime(item[0], DATE_FORMAT))
    return [json.loads(content) for _, content in timestamped_requests]


def read_packaged_requests(dataset: str) -> List[Dict]:
    try:
        data_package = resources.files("ai_infra_bench_dataset")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The packaged datasets are unavailable; install ai-infra-bench[data]"
        ) from error

    data_directory = data_package.joinpath("data", dataset)
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"Payload dataset {dataset!r} is not included in ai-infra-bench-dataset"
        )

    payload_resource = data_directory.joinpath("payload.jsonl")
    shard_resources = sorted(
        (
            resource
            for resource in data_directory.iterdir()
            if resource.is_file()
            and resource.name.startswith("payload-")
            and resource.name.endswith(".jsonl")
            and resource.name[len("payload-") : -len(".jsonl")].isdigit()
        ),
        key=lambda resource: resource.name,
    )
    payload_resources = shard_resources or (
        [payload_resource] if payload_resource.is_file() else []
    )
    if not payload_resources:
        raise FileNotFoundError(
            f"Payload dataset {dataset!r} is not included in ai-infra-bench-dataset"
        )

    requests: List[Dict] = []
    for payload_resource in payload_resources:
        with resources.as_file(payload_resource) as payload_path:
            requests.extend(read_requests(str(Path(payload_path))))
    return requests
