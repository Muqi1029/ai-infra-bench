import logging
import os
import shutil
from datetime import datetime
from typing import List

from ai_infra_bench.constants import FULL_DATA_JSON_PATH
from ai_infra_bench.utils.ori import is_ci

logger = logging.getLogger(__name__)


def check_dir(output_dir: str, full_data_json_path: str = FULL_DATA_JSON_PATH) -> str:
    if is_ci():
        os.makedirs(os.path.join(output_dir, full_data_json_path), exist_ok=True)
        return output_dir

    if os.path.exists(output_dir):
        while True:
            option = input(
                f"Directory '{output_dir}' exists. Choose [1] delete, [2] rename, "
                "[3] new path, [4] quit: "
            ).strip()
            if option == "1":
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                break
            if option == "2":
                output_dir = f"{output_dir}_{datetime.now():%m%d_%H%M}"
                os.makedirs(output_dir)
                break
            if option == "3":
                output_dir = input("New directory name: ").strip()
                os.makedirs(output_dir)
                break
            if option == "4":
                raise SystemExit(0)
            logger.warning("Invalid option. Choose 1, 2, 3, or 4.")
    else:
        os.makedirs(output_dir)

    os.makedirs(os.path.join(output_dir, full_data_json_path), exist_ok=True)
    return output_dir


def check_values_in_features_metrics(
    input_features: List[str], output_metrics: List[str]
) -> None:
    values = [*input_features, *output_metrics]
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(
            "input_features and output_metrics must contain non-empty strings"
        )


def check_param_in_cmd(param: str, cmds: List[str]) -> None:
    for cmd in cmds:
        if param in cmd:
            raise ValueError(f"{cmd=} should not contain '{param}'")


def check_str_list_str(cmds: str | List[str]) -> List[str]:
    if isinstance(cmds, str):
        return [cmds]
    if isinstance(cmds, list) and all(isinstance(cmd, str) for cmd in cmds):
        return cmds
    raise ValueError(f"cmds must be str or List[str], got {cmds=}")


def check_client_labels(
    client_labels: None | str | List[str] | List[List[str]],
    num_clients: List[int],
) -> List[None | List[str]]:
    if client_labels is None:
        return [None] * len(num_clients)
    if isinstance(client_labels, str):
        labels = [[client_labels] * count for count in num_clients]
    elif isinstance(client_labels, list) and all(
        isinstance(label, str) for label in client_labels
    ):
        if len(client_labels) != num_clients[0]:
            raise ValueError("client_labels length does not match the client count")
        labels = [client_labels]
    elif isinstance(client_labels, list) and all(
        isinstance(group, list) for group in client_labels
    ):
        labels = client_labels
    else:
        raise TypeError(
            "client_labels must be None, str, list[str], or list[list[str]]"
        )

    if len(labels) != len(num_clients) or any(
        len(group) != count for group, count in zip(labels, num_clients)
    ):
        raise ValueError("client_labels do not match the client counts")
    if not all(isinstance(label, str) for group in labels for label in group):
        raise TypeError("client labels must be strings")
    return labels


def check_server_labels(
    server_labels: None | str | List[str], num_servers: int
) -> List[str | None]:
    if server_labels is None:
        return [None] * num_servers
    if isinstance(server_labels, str):
        return [server_labels] * num_servers
    if isinstance(server_labels, list) and len(server_labels) == num_servers:
        if not all(isinstance(label, str) for label in server_labels):
            raise TypeError("server labels must be strings")
        return server_labels
    raise ValueError("server_labels must be None, a string, or one label per server")
