import os
import shutil
from datetime import datetime
from typing import List, Union

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
                exit(0)
            else:
                print("Invalid option. Please enter '1', '2' or '3'.")
    else:
        # If the directory does not exist, create it directly
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    os.makedirs(os.path.join(output_dir, full_data_json_path))
    return output_dir


def check_output_file(cmds: Union[List[str], str]):
    if isinstance(cmds, str):
        cmds = [cmds]

    for cmd in cmds:
        assert (
            "--output-file" not in cmd
        ), f"{cmd=} should not use --output-file, it will be generated automatically"


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
    check_output_file(client_cmds)

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


def check_param_in_cmd(param: str, cmds: List[str]):
    for cmd in cmds:
        assert param not in cmd, f"{cmd=} should not contain '{param}''"


def slo_check_params(server_cmds, client_cmds, labels):
    check_server_client_cmds(server_cmds, client_cmds, labels=labels)
    assert len(server_cmds) == len(
        client_cmds
    ), f"The length os server_cmds and client_cmds should be equal, but found {len(server_cmds)=}, {len(client_cmds)=}"

    assert all(
        "request-rate" not in cmd for cmd in client_cmds
    ), "request-rate should not be set in the client_cmds"
    assert all(
        "max-concurrency" not in cmd for cmd in client_cmds
    ), "max-concurrency should not be set in the client_cmds"
