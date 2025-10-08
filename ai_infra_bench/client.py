import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

from ai_infra_bench.check import (
    check_client_labels,
    check_dir,
    check_param_in_cmd,
    check_server_labels,
    check_values_in_features_metrics,
)
from ai_infra_bench.modes.gen import gen_export_csv, gen_export_table, gen_plot, gen_run
from ai_infra_bench.modes.slo import slo_run
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    add_request_rate,
    kill_process_tree,
    maybe_create_labels,
    maybe_warmup,
)

logger = logging.getLogger(__name__)


def client_slo(
    client_cmds: List[str],
    input_features: List[str],
    output_metrics: List[str],
    check_slo: Callable | List[Callable],
    request_rates: List[Tuple[int, int]] | Tuple[int, int],
    labels: Optional[List[str]] = None,
    n=1,
    output_dir="output",
    disable_warmup=False,
    disable_table=False,
    disable_csv=False,
):
    if isinstance(client_cmds, str):
        client_cmds = [client_cmds]
    if isinstance(check_slo, Callable):
        check_slo = [check_slo] * len(client_cmds)
    if isinstance(request_rates, tuple):
        request_rates = [request_rates]

    check_values_in_features_metrics(input_features, output_metrics)
    check_param_in_cmd("output-file", client_cmds)
    check_param_in_cmd("request-rate", client_cmds)
    check_param_in_cmd("max-concurrency", client_cmds)
    assert (
        len(client_cmds) == len(request_rates) == len(check_slo)
    ), "Length of client_cmds, request_rates, and check_slo must be the same"

    labels = maybe_create_labels(labels, len(client_cmds))

    output_dir = check_dir(output_dir, FULL_DATA_JSON_PATH)

    try:
        all_clients_results: List[List[Dict]] = []
        answers = []
        for client_cmd, request_rate, check_slo, label in zip(
            client_cmds, request_rates, check_slo, labels
        ):
            maybe_warmup(
                add_request_rate(client_cmd, request_rate[0]),
                output_dir=output_dir,
                disable_warmup=disable_warmup,
            )
            client_results, answer = slo_run(
                client_cmd=client_cmd,
                request_rate=request_rate,
                check_slo=check_slo,
                n=n,
                output_dir=output_dir,
                label=label,
            )
            all_clients_results.extend(client_results)
            answers.append(answer)

        if not disable_table:
            gen_export_table(
                all_clients_results=all_clients_results,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
            )
        if not disable_csv:
            gen_export_csv(
                all_clients_results=all_clients_results,
                output_dir=output_dir,
                labels=labels,
            )

    except Exception as e:
        kill_process_tree(os.getpid(), include_parent=False)
        raise RuntimeError(f"Process failed with error: {e}") from e


def client_gen(
    client_cmds: str | List[str],
    *,
    input_features: List[str],
    output_metrics: List[str],
    server_label: None | str = None,
    client_labels: None | str | List[str] = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
):
    if isinstance(client_cmds, str):
        client_cmds = [client_cmds]
    if not (isinstance(client_cmds, list) and isinstance(client_cmds[0], str)):
        raise ValueError(
            f"client_cmds must be a string or a list of strings (for multiple clients), but found {client_cmds=}"
        )

    assert (
        isinstance(client_labels, list)
        and all(isinstance(label, str) for label in client_labels)
        and len(client_labels) == len(client_cmds)
    ), "client_labels should be a list of strings"

    check_values_in_features_metrics(input_features, output_metrics)
    check_param_in_cmd("output-file", client_cmds)
    server_labels = check_server_labels(server_label, 1)
    client_labels = check_client_labels(
        client_labels=client_labels, num_clients=[len(client_cmds)]
    )

    output_dir = check_dir(output_dir, FULL_DATA_JSON_PATH)

    try:
        maybe_warmup(
            cmd=client_cmds[0], output_dir=output_dir, disable_warmup=disable_warmup
        )

        all_clients_results: List[List[Dict]] = gen_run(
            client_cmds=client_cmds,
            n=n,
            labels=maybe_create_labels(
                num=len(client_cmds),
                server_label=server_labels[0],
                client_labels=client_labels[0],
            ),
            output_dir=output_dir,
        )

        if not disable_table:
            gen_export_table(
                all_clients_results=all_clients_results,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
            )

        if not disable_csv:
            gen_export_csv(
                all_clients_results=all_clients_results, output_dir=output_dir
            )

        if not disable_plot:
            gen_plot(
                all_clients_results=all_clients_results,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
            )
    except Exception as e:
        kill_process_tree(os.getpid(), include_parent=False)
        raise RuntimeError(f"Process failed with error: {e}") from e
