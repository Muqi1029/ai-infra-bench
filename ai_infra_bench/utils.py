from __future__ import annotations

import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, List

import numpy as np
import psutil
import requests


@dataclass
class ServerAccessInfo:
    base_url: str
    api_key: str = None


logger = logging.getLogger(__name__)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
graph_per_row = 3
FULL_DATA_JSON_PATH = "full_data_json"  # used to store all json files
TABLE_NAME = "table.md"
CSV_NAME = "data.csv"
WARMUP_FILE = ".warmup.json"
demo_output = {
    "backend": "sglang-oai",
    "dataset_name": "random",
    "request_rate": 10.0,
    "max_concurrency": 10,
    "sharegpt_output_len": None,
    "random_input_len": 1200,
    "random_output_len": 800,
    "random_range_ratio": 1.0,
    "duration": 45.11868940386921,
    "completed": 100,
    "total_input_tokens": 120000,
    "total_output_tokens": 80000,
    "total_output_tokens_retokenized": 79998,
    "request_throughput": 2.2163764356024127,
    "input_throughput": 2659.6517227228956,
    "output_throughput": 1773.1011484819303,
    "mean_e2e_latency_ms": 4482.026166650467,
    "median_e2e_latency_ms": 4487.435979535803,
    "std_e2e_latency_ms": 32.15524448450066,
    "p99_e2e_latency_ms": 4534.823208898306,
    "mean_ttft_ms": 38.534140698611736,
    "median_ttft_ms": 42.44273528456688,
    "std_ttft_ms": 10.558202315257851,
    "p99_ttft_ms": 61.15902605932206,
    "mean_tpot_ms": 5.561316678287678,
    "median_tpot_ms": 5.56157646876747,
    "std_tpot_ms": 0.04168330778296244,
    "p99_tpot_ms": 5.627061070545631,
    "mean_itl_ms": 5.561935330397016,
    "median_itl_ms": 5.495080258697271,
    "std_itl_ms": 1.1977701758121588,
    "p95_itl_ms": 6.047771545127034,
    "p99_itl_ms": 6.62423954345286,
    "concurrency": 9.933857179517508,
    "accept_length": None,
}


def cmp_preprocess_client_cmds(
    client_cmds: List[str], server_access_info: ServerAccessInfo
) -> List[str]:
    api_key = server_access_info.api_key
    if api_key is not None:
        if api_key.startswith("Bearer"):
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["API_KEY"] = api_key

    return [cmd + f" --base-url {server_access_info.base_url}" for cmd in client_cmds]


def enter_decorate(title: str, filename: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[{title}] Start...")

            result = func(*args, **kwargs)

            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            output_dir = bound.arguments.get("output_dir", "")
            if output_dir:
                logger.info(
                    f"[{title}] SUCCESS: {filename} has been successfully generated in {output_dir}"
                )
            else:
                logger.info(f"[{title}] SUCCESS.")
            return result

        return wrapper

    return decorator


def is_ci() -> bool:
    return any(
        os.getenv(var, "").lower() == "true"
        for var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]
    )


def maybe_create_labels(
    num_clients: int,
    server_label: None | str = None,
    client_labels: List[str] | None = None,
) -> List[str]:
    """
    Receive one group of server_label and client_labels:

    if both server_label and client_labels are None, default [client01, client02, ...], whose length is num_client
    else repeat server_label with num_clients times or use check_labels directly
    """

    def _validate_client_labels(labels: List[str]):
        if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
            raise TypeError(f"client_labels must be a list of strings, got {labels!r}")
        if len(labels) != num_clients:
            raise ValueError(f"Expected {num_clients} client labels, got {len(labels)}")

    if server_label is None:
        if client_labels is None:
            labels = [f"client{i:02d}" for i in range(num_clients)]
            logger.info(f"Auto-generated labels: {labels}")
            return labels
        else:
            _validate_client_labels(client_labels)
            return client_labels

    labels = [server_label] * num_clients
    if client_labels is not None:
        _validate_client_labels(client_labels)
        labels = [
            f"{label}_{client_label}"
            for label, client_label in zip(labels, client_labels)
        ]
    return labels


def maybe_warmup(cmd: str, output_dir: str, disable_warmup: bool):
    if disable_warmup:
        logger.info("Warmup is disabled.")
        return
    logger.info("Starting warmup...")

    output_file = os.path.join(output_dir, WARMUP_FILE)
    cmd += f" --output-file {output_file}"
    run_cmd(cmd, is_block=True)
    if is_ci():
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(demo_output, f)


def wait_for_server(base_url: str, timeout=None):
    if is_ci():
        return
    start_time = time.perf_counter()

    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models", headers={"Authorization": "Muqi1029"}
            )
            if response.status_code == 200:
                print("Server becomes ready!")
                break
            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError(
                    "Server did not become ready within the timeout period"
                )
        except requests.exceptions.RequestException:
            time.sleep(1)


def run_cmd(cmd: str, is_block=True):
    if is_ci():
        logger.info("[CI mode] Skipping real GPU execution.")
        return

    cmd = cmd.replace("\\\n", " ").replace("\\", " ")
    if is_block:
        return subprocess.run(cmd.split(), text=True, stderr=subprocess.STDOUT)
    else:
        return subprocess.Popen(cmd.split(), text=True, stderr=subprocess.STDOUT)


def read_jsonl(filepath: str):
    if is_ci():
        return [demo_output]
    data = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def avg_std_strf(
    key: str, item_list: List[Dict[str, Any]], *, sep=", ", precision: int = None
) -> str:
    val_list = [item[key] for item in item_list]

    fmt = "" if precision is None else f".{precision}f"

    if not isinstance(val_list[0], (int, float)):
        return str(val_list[0])

    if len(val_list) == 1 or (std := np.std(val_list, ddof=1)) == 0:
        return format(val_list[0], fmt)

    avg = np.mean(val_list)
    return (
        f"{format(avg, fmt)} \u00b1 {format(std, fmt)}"
        f"({sep.join(format(val, fmt) for val in val_list)})"
    )


def add_request_rate(cmd: str, rate: int):
    cmd += f" --max-concurrency {rate} --request-rate {rate}"
    if "num-prompt" not in cmd:
        cmd += f" --num-prompt {min(max(100, rate * 10), 250)}"
    return cmd


def sort_data_by_key(key: str, data: List[List[Dict]]):
    num_points = len(data)
    if num_points == 0:
        return data
    assert isinstance(data[0][0][key], (int, float))
    val_list = [item_list[0][key] for item_list in data]
    sorted_indices = sorted(range(len(data)), key=lambda i: val_list[i])
    sorted_data = []
    for idx in sorted_indices:
        sorted_data.append(data[idx])
    return sorted_data


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass
