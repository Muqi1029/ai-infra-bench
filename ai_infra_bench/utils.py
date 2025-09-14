from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List

import numpy as np
import psutil
import requests

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
graph_per_row = 3
FULL_DATA_JSON_PATH = "full_data_json"  # used to store all json files


def warmup(cmd: str, output_dir: str):
    cmd += f" --output-file {output_dir}/.warmup.json"
    run_cmd(cmd, is_block=True)


def wait_for_server(base_url: str, timeout=None):
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
    cmd = cmd.replace("\\\n", " ").replace("\\", " ")
    if is_block:
        return subprocess.run(cmd.split(), text=True, stderr=subprocess.STDOUT)
    return subprocess.Popen(cmd.split(), text=True, stderr=subprocess.STDOUT)


def dummy_get_filename(i, label):
    return f"{label}_client_{i:02d}.jsonl"


def read_jsonl(filepath: str):
    data = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def avg_std_strf(
    key: str, item_list: List[Dict[str, float]], *, sep=", ", precision: int = None
) -> str:
    val_list = [item[key] for item in item_list]

    if isinstance(val_list[0], str):
        return val_list[0]

    fmt = "" if precision is None else f".{precision}f"

    if len(val_list) == 1:
        return format(val_list[0], fmt)

    avg = np.mean(val_list)
    std = np.std(val_list, ddof=1)

    return (
        f"{format(avg, fmt)} \u00b1 {format(std, fmt)} "
        f"({sep.join(format(val, fmt) for val in val_list)})"
    )


def add_request_rate(cmd: str, rate: int):
    cmd += f" --max-concurrency {rate} --request-rate {rate}"
    if "num-prompt" not in cmd:
        cmd += f" --num-prompt {rate * 10}"
    return cmd


def sort_data_by_key(key: str, data: List[List[Dict]]):
    num_points = len(data)
    if num_points == 0:
        return data
    assert isinstance(data[0][key], (int, float))
    val_list = [item_list[0][key] for item_list in data]
    sorted_indices = sorted(range(range(len(data))), key=lambda i: val_list[i])
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
