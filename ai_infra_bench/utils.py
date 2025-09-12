from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil
import requests

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
graph_per_row = 3

SGLANG_KEYS = [
    "backend",
    "dataset_name",
    "request_rate",
    "max_concurrency",
    "sharegpt_output_len",
    "random_input_len",
    "random_output_len",
    "random_range _ratio",
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "total_output_tokens_retokenized",
    "request_throughput",
    "input_through put",
    "output_throughput",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "std_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms ",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p95_it l_ms",
    "p99_itl_ms",
    "concurrency",
    "accept_length",
]


def check_dir(output_dir: str, full_data_json_path):
    """
    Checks if the specified output directory exists. If it does, it prompts the user
    for an action (delete or rename). It re-prompts on invalid input.
    """
    if os.path.exists(output_dir):
        while True:
            # Re-prompt loop
            prompt_text = (
                f"The directory '{output_dir}' already exists. Please choose an option:\n"
                "  1. Delete the existing directory and create a new one.\n"
                "  2. Append a timestamp to the directory name (e.g., 'your_dir_MMDD_HHMM').\n"
                "  3. Quit.\n"
                "Enter your choice (1, 2 or 3): "
            )
            option = input(prompt_text).strip()

            if option == "1":
                print(f"Deleting '{output_dir}'...")
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                print(f"Directory '{output_dir}' created.")
                break
            elif option == "2":
                date_suffix = datetime.now().strftime("%m%d_%H%M")
                output_dir = f"{output_dir}_{date_suffix}"
                os.makedirs(output_dir)
                print(f"New directory created: '{output_dir}'.")
                break
            if option == "3":
                break
            else:
                print("Invalid option. Please enter '1', '2' or '3'.")
    else:
        # If the directory does not exist, create it directly
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    os.makedirs(os.path.join(output_dir, full_data_json_path))


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


def check_input_features_metrics(input_features, metrics):
    for input_feature in input_features:
        assert (
            input_feature in SGLANG_KEYS
        ), f"{input_feature=} should be in the {SGLANG_KEYS=}"

    for metric in metrics:
        assert metric in SGLANG_KEYS, f"{metric=} should be all in the {SGLANG_KEYS=}"


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
