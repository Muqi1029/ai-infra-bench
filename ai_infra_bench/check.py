import logging
import os
import shutil
from datetime import datetime
from typing import List

from ai_infra_bench.constants import DEFAULT_BENCH_SERVING_PATH, SGLANG_KEYS
from ai_infra_bench.utils import is_ci

try:
    import sglang

    is_sglang_available = True
except ImportError:
    is_sglang_available = False

logger = logging.getLogger(__name__)


def ensure_bench_serving_available() -> None:
    """
    Automatically download bench_serving.py if sglang is not available.
    Also installs required dependencies.
    """
    if is_sglang_available:
        return

    if os.path.exists(DEFAULT_BENCH_SERVING_PATH):
        logger.info(f"bench_serving.py already exists at {DEFAULT_BENCH_SERVING_PATH}")
        return

    logger.info("sglang is not available, downloading bench_serving.py...")
    try:
        import requests

        raw_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/bench_serving.py"
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()

        os.makedirs(os.path.dirname(DEFAULT_BENCH_SERVING_PATH), exist_ok=True)
        with open(DEFAULT_BENCH_SERVING_PATH, "w") as f:
            f.write(response.text)
        logger.info(
            f"Successfully downloaded bench_serving.py to {DEFAULT_BENCH_SERVING_PATH}"
        )

        # Install dependencies required by bench_serving.py
        install_bench_serving_dependencies()

    except Exception as e:
        logger.error(
            f"Failed to download bench_serving.py from {raw_url}: {e}. "
            f"Please ensure you have internet access or manually download the file."
        )
        raise


def install_bench_serving_dependencies() -> None:
    """
    Parse bench_serving.py and install missing dependencies.
    """
    import ast
    import subprocess
    import sys

    logger.info("Installing bench_serving.py dependencies...")

    try:
        with open(DEFAULT_BENCH_SERVING_PATH, "r") as f:
            tree = ast.parse(f.read())

        # Extract all imported modules
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        # Standard library modules to skip
        stdlib_modules = set(sys.stdlib_module_names)

        # Filter out standard library modules
        third_party_imports = {
            mod for mod in imports if mod not in stdlib_modules and mod != "sglang"
        }

        if not third_party_imports:
            logger.info("No third-party dependencies found to install")
            return

        logger.info(
            "Found dependencies to install: %s",
            ", ".join(sorted(third_party_imports)),
        )

        # Try to install each dependency
        for module in sorted(third_party_imports):
            try:
                __import__(module)
                logger.debug(f"  {module} is already installed")
            except ImportError:
                # Map module names to pip package names (handle common cases)
                package_map = {
                    "cv2": "opencv-python",
                    "PIL": "Pillow",
                    "yaml": "PyYAML",
                    "bs4": "beautifulsoup4",
                }
                package_name = package_map.get(module, module)

                logger.info("  Installing %s...", package_name)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.debug(f"  {package_name} installed successfully")

        logger.info("All dependencies installed successfully")

    except Exception as e:
        logger.warning(
            f"Failed to automatically install dependencies: {e}. "
            f"Please manually install any missing dependencies if the script fails."
        )


def check_dir(output_dir: str, full_data_json_path):
    """
    Checks if the specified output directory exists. If it does, it prompts the user
    for an action (delete or rename). It re-prompts on invalid input.
    """
    if is_ci():
        os.makedirs(os.path.join(output_dir, full_data_json_path), exist_ok=True)
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
                break
            elif option == "4":
                exit(0)
            else:
                logger.warning("Invalid option. Please enter '1', '2', '3', or '4'.")
    else:
        # If the directory does not exist, create it directly
        os.makedirs(output_dir)
        logger.info(f"Directory '{output_dir}' created.")
    os.makedirs(os.path.join(output_dir, full_data_json_path))
    logger.info(f"output_dir set to '{output_dir}'")
    return output_dir


def check_content_client_cmds(client_cmds: List[List[str]]) -> None:
    if is_sglang_available:
        for client_cmd in client_cmds:
            for cmd in client_cmd:
                assert any(
                    cmd.strip().startswith(p)
                    for p in [
                        "python -m sglang.bench_serving",
                        "python3 -m sglang.bench_serving",
                    ]
                ), f"Each client_cmd must start with 'python -m sglang.bench_serving' or 'python3 -m sglang.bench_serving', but found {cmd=}"
    else:
        # Ensure bench_serving is available if sglang is not installed
        ensure_bench_serving_available()
        for cmd_list_idx, client_cmd in enumerate(client_cmds):
            for cmd_idx, cmd in enumerate(client_cmd):
                if cmd.startswith("python -m sglang.bench_serving"):
                    cmd = cmd.replace(
                        "python -m sglang.bench_serving",
                        f"python {DEFAULT_BENCH_SERVING_PATH}",
                    )
                elif cmd.startswith("python3 -m sglang.bench_serving"):
                    cmd = cmd.replace(
                        "python3 -m sglang.bench_serving",
                        f"python3 {DEFAULT_BENCH_SERVING_PATH}",
                    )
                else:
                    raise ValueError(
                        f"Each client_cmd must start with 'python -m sglang.bench_serving' or 'python3 -m sglang.bench_serving', but found {cmd=}"
                    )
                client_cmd[cmd_idx] = cmd
            client_cmds[cmd_list_idx] = client_cmd


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
        if is_sglang_available:
            for cmd in client_cmd:
                assert any(
                    cmd.strip().startswith(p)
                    for p in [
                        "python -m sglang.bench_serving",
                        "python3 -m sglang.bench_serving",
                    ]
                ), f"Each client_cmd must start with 'python -m sglang.bench_serving' or 'python3 -m sglang.bench_serving', but found {cmd=}"
        else:
            # Ensure bench_serving is available if sglang is not installed
            ensure_bench_serving_available()
            for cmd_idx, cmd in enumerate(client_cmd):
                if cmd.startswith("python -m sglang.bench_serving"):
                    cmd = cmd.replace(
                        "python -m sglang.bench_serving",
                        f"python {DEFAULT_BENCH_SERVING_PATH}",
                    )
                elif cmd.startswith("python3 -m sglang.bench_serving"):
                    cmd = cmd.replace(
                        "python3 -m sglang.bench_serving",
                        f"python3 {DEFAULT_BENCH_SERVING_PATH}",
                    )
                else:
                    raise ValueError(
                        f"Each client_cmd must start with 'python -m sglang.bench_serving' or 'python3 -m sglang.bench_serving', but found {cmd=}"
                    )
                client_cmd[cmd_idx] = cmd


def check_values_in_features_metrics(input_features, output_metrics):
    for input_feature in input_features:
        assert (
            input_feature in SGLANG_KEYS
        ), f"{input_feature=} should be in the {SGLANG_KEYS=}"

    for metric in output_metrics:
        assert metric in SGLANG_KEYS, f"{metric=} should be in the {SGLANG_KEYS=}"


def check_param_in_cmd(param: str, cmds: List[str]):
    for cmd in cmds:
        assert param not in cmd, f"{cmd=} should not contain '{param}'"


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
