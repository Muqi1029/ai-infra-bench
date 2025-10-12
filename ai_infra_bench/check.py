import logging
import os
import shutil
from datetime import datetime
from typing import List

from ai_infra_bench.utils import is_ci

logger = logging.getLogger(__name__)

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
    if is_ci():
        os.makedirs(os.path.join(output_dir, full_data_json_path))
        return output_dir

    if os.path.exists(output_dir):
        while True:
            # Re-prompt loop
            prompt_text = (
                f"The directory '{output_dir}' already exists. Please choose an option:\n"
                "  1. Delete the existing directory and create a new one.\n"
                "  2. Append a timestamp to the directory name (e.g., 'your_dir_MMDD_HHMM').\n"
                "  3. Input a new directory name.\n"
                "  4. Quit.\n"
                "Enter your choice (1, 2, 3 or 4): "
            )
            option = input(prompt_text).strip()

            if option == "1":
                logger.info(f"Deleting '{output_dir}'...")
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                logger.info(f"Directory '{output_dir}' created.")
                break
            elif option == "2":
                date_suffix = datetime.now().strftime("%m%d_%H%M")
                output_dir = f"{output_dir}_{date_suffix}"
                os.makedirs(output_dir)
                logger.info(f"New directory created: '{output_dir}'.")
                break
            elif option == "3":
                output_dir = input("New directory name: ").strip()
                os.makedirs(output_dir)
                logger.info(f"New directory created: '{output_dir}'.")
            elif option == "4":
                exit(0)
            else:
                logger.warning("Invalid option. Please enter '1', '2' or '3'.")
    else:
        # If the directory does not exist, create it directly
        os.makedirs(output_dir)
        logger.info(f"Directory '{output_dir}' created.")
    os.makedirs(os.path.join(output_dir, full_data_json_path))
    logger.info(f"output_dir set to '{output_dir}'")
    return output_dir


def check_content_server_client_cmds(
    server_cmds: List[str], client_cmds: List[List[str]]
) -> None:
    for cmd in server_cmds:
        assert any(
            cmd.strip().startswith(p)
            for p in [
                "python -m sglang.launch_server",
                "python3 -m sglang.launch_server",
            ]
        ), f"Each server_cmd must start with 'python -m sglang.launch_server' or 'python3 -m sglang.launch_server', but found {cmd=}"

    for client_cmd in client_cmds:
        for cmd in client_cmd:
            assert any(
                cmd.strip().startswith(p)
                for p in [
                    "python -m sglang.bench_serving",
                    "python3 -m sglang.bench_serving",
                ]
            ), f"Each client_cmd must start with 'python -m sglang.bench_serving' or 'python3 -m sglang.bench_serving', but found {cmd=}"


def check_values_in_features_metrics(input_features, output_metrics):
    for input_feature in input_features:
        assert (
            input_feature in SGLANG_KEYS
        ), f"{input_feature=} should be in the {SGLANG_KEYS=}"

    for metric in output_metrics:
        assert metric in SGLANG_KEYS, f"{metric=} should be in the {SGLANG_KEYS=}"


def check_param_in_cmd(param: str, cmds: List[str]):
    for cmd in cmds:
        assert param not in cmd, f"{cmd=} should not contain '{param}''"


def check_str_list_str(cmds: str | List[str]):
    if isinstance(cmds, str):
        cmds = [cmds]
    elif not (isinstance(cmds, list) and all(isinstance(cmd, str) for cmd in cmds)):
        raise ValueError(f"cmds must be str or List[str], got {cmds=}")
    return cmds


def check_client_labels(
    client_labels: None | str | List[str] | List[List[str]],
    num_clients: List[int],
) -> List[None | List[str]]:
    """
    Normalize and validate client labels for multiple servers.

    This function ensures that `client_labels` is represented as a list of lists,
    where each sublist corresponds to the labels of clients under one server.

    Supported input formats:
      - None: returns `[None] * len(num_clients)`
      - str: a single label is repeated for each client under each server
      - list[str]: a single server's labels; must match `num_clients[0]`
      - list[list[str]]: explicit labels per server and client
    """
    if client_labels is None:
        return [None] * len(num_clients)

    if isinstance(client_labels, str):
        client_labels = [[client_labels] * n for n in num_clients]
    elif isinstance(client_labels, list):
        if all(isinstance(label, str) for label in client_labels):
            # list[str]
            assert len(client_labels) == num_clients[0]
            client_labels = [client_labels]
        elif all(isinstance(label, list) for label in client_labels):
            # list[list[str]]
            assert all(
                isinstance(label, str)
                for client_label_list in client_labels
                for label in client_label_list
            )
        else:
            raise TypeError("client_labels list must contain only str or list[str].")
    else:
        raise TypeError(
            f"client_labels must be None, str, list[str], or list[list[str]], "
            f"but got {type(client_labels).__name__}."
        )

    assert len(client_labels) == len(num_clients)
    for idx in range(len(client_labels)):
        assert (
            len(client_labels[idx]) == num_clients[idx]
        ), f"Found {len(client_labels[idx])=} {num_clients[idx]=}"
    return client_labels


def check_server_labels(
    server_labels: None | str | List[str],
    num_servers: int,
) -> List[str | None]:
    """
     Normalize and validate server labels for multiple servers.

    This function ensures that `server_labels` is a list of length `num_servers`.
    - If `server_labels` is None, it returns a list of `[None] * num_servers`.
    - If it is a single string, it repeats that string `num_servers` times.
    - If it is a list of strings, it checks that its length matches `num_servers`
      and that all elements are strings.
    """
    # Case 0: None
    if server_labels is None:
        return [None] * num_servers

    # Case 1: single string
    if isinstance(server_labels, str):
        return [server_labels] * num_servers

    # Case 2: list of strings
    if isinstance(server_labels, list):
        if len(server_labels) != num_servers:
            raise ValueError(
                f"Expected {num_servers} server labels, got {len(server_labels)}."
            )
        if not all(isinstance(label, str) for label in server_labels):
            raise TypeError("All server_labels must be strings.")
        return server_labels

    # Case 3: invalid type
    raise TypeError(
        f"server_labels must be None, str, or list[str], got {type(server_labels).__name__}."
    )
