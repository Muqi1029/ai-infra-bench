from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

import psutil
import requests

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
graph_per_row = 3


def warmup(cmd: str, output_dir: str):
    cmd += f" --output-file {output_dir}/.warmup.json"
    run_cmd(cmd, is_block=True)


def check_server_client_cmds(server_cmds, client_cmds, *, labels):
    assert all(
        [
            cmd.strip().startswith("python -m sglang.launch_server")
            for cmd in server_cmds
        ]
    ), "Each server_cmd must startswith 'python -m sglang.launch_server'"

    if isinstance(client_cmds[0], list):
        for client_cmd in client_cmds:
            assert all(
                [
                    cmd.strip().startswith("python -m sglang.bench_serving")
                    for cmd in client_cmd
                ]
            ), "Each client_cmd must start with 'python -m sglang.bench_serving'"
    elif isinstance(client_cmds[0], str):
        assert all(
            [
                cmd.strip().startswith("python -m sglang.bench_serving")
                for cmd in client_cmds
            ]
        ), "Each client_cmd must start with 'python -m sglang.bench_serving'"

    # FIXME(muqi1029): don't let the user set output_file
    assert all(
        ["output-file" not in cmd for cmd in server_cmds]
    ), "Set output-file is not supported yet"

    assert len(server_cmds) == len(
        labels
    ), f"The length of server_cmds and labels should be equal, but found {len(server_cmds)=}, {len(labels)=}"
    # TODO: check metrics, check_slo


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
    return f"{label}_client_{i}.jsonl"


def read_jsonl(filepath: str):
    data = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
