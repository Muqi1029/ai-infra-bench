import logging
import os
from typing import Callable, Dict, List, Tuple

from ai_infra_bench.check import (
    check_client_labels,
    check_dir,
    check_param_in_cmd,
    check_server_labels,
    check_str_list_str,
    check_values_in_features_metrics,
)
from ai_infra_bench.modes.cmp import cmp_export_table
from ai_infra_bench.modes.gen import gen_export_csv, gen_export_table, gen_plot, gen_run
from ai_infra_bench.modes.slo import slo_run
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    ServerAccessInfo,
    add_request_rate,
    cmp_preprocess_client_cmds,
    kill_process_tree,
    maybe_create_labels,
    maybe_warmup,
)

logger = logging.getLogger(__name__)


def client_slo(
    client_cmds: str | List[str],
    *,
    input_features: List[str],
    output_metrics: List[str],
    check_slo: Callable | List[Callable],
    request_rates: Tuple[int, int] | List[Tuple[int, int]],
    server_label: None | str = None,
    client_labels: None | str | List[str] = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
) -> None:
    client_cmds = check_str_list_str(client_cmds)

    if isinstance(check_slo, Callable):
        check_slo = [check_slo] * len(client_cmds)
    if isinstance(request_rates, tuple):
        request_rates = [request_rates] * len(client_cmds)

    server_label = check_server_labels(server_labels=server_label, num_servers=1)[0]
    client_labels = check_client_labels(
        client_labels=client_labels, num_clients=[len(client_cmds)]
    )[0]

    check_values_in_features_metrics(
        input_features=input_features, output_metrics=output_metrics
    )
    check_param_in_cmd("output-file", client_cmds)
    check_param_in_cmd("request-rate", client_cmds)
    check_param_in_cmd("max-concurrency", client_cmds)
    assert (
        len(client_cmds) == len(request_rates) == len(check_slo)
    ), f"Length of client_cmds, request_rates, and check_slo must be the same, but {len(client_cmds)=}, {len(request_rates)=} {len(check_slo)=}"

    output_dir = check_dir(output_dir, FULL_DATA_JSON_PATH)

    try:
        all_clients_results: List[List[Dict]] = []
        answers = []

        labels = maybe_create_labels(
            num_clients=len(client_cmds),
            server_label=server_label,
            client_labels=client_labels,
        )
        for idx, (client_cmd, request_rate, check_slo_func) in enumerate(
            zip(client_cmds, request_rates, check_slo)
        ):
            maybe_warmup(
                cmd=add_request_rate(client_cmd, request_rate[0]),
                output_dir=output_dir,
                disable_warmup=disable_warmup,
            )
            client_results, answer = slo_run(
                client_cmd=client_cmd,
                request_rate=request_rate,
                check_slo=check_slo_func,
                n=n,
                output_dir=output_dir,
                label=labels[idx],
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
) -> None:
    client_cmds = check_str_list_str(client_cmds)
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
        labels = maybe_create_labels(
            num_clients=len(client_cmds),
            server_label=server_labels[0],
            client_labels=client_labels[0],
        )

        all_clients_results: List[List[Dict]] = gen_run(
            client_cmds=client_cmds,
            n=n,
            labels=labels,
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


def client_cmp(
    server_access_info: ServerAccessInfo | List[ServerAccessInfo],
    client_cmds: str | List[str],
    *,
    input_features: List[str],
    output_metrics: List[str],
    server_labels: None | List[str] = None,
    client_labels: None | List[str] = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
) -> None:
    if isinstance(server_access_info, ServerAccessInfo):
        server_access_info = [server_access_info]

    client_cmds = check_str_list_str(client_cmds)
    check_values_in_features_metrics(input_features, output_metrics)
    check_param_in_cmd("output-file", client_cmds)
    check_param_in_cmd("base-url", client_cmds)
    check_param_in_cmd("host", client_cmds)
    check_param_in_cmd("port", client_cmds)

    num_servers = len(server_access_info)
    num_clients = len(client_cmds)

    server_labels = check_server_labels(
        server_labels=server_labels, num_servers=num_servers
    )
    client_labels = check_client_labels(client_labels, [num_clients])[0]

    output_dir = check_dir(
        output_dir=output_dir, full_data_json_path=FULL_DATA_JSON_PATH
    )
    try:
        all_clients_results: List[List[Dict]] = []
        for i in range(num_servers):
            cmp_client_cmds = cmp_preprocess_client_cmds(
                client_cmds, server_access_info[i]
            )
            maybe_warmup(
                cmd=cmp_client_cmds[0],
                output_dir=output_dir,
                disable_warmup=disable_warmup,
            )
            labels = maybe_create_labels(
                num_clients=num_clients,
                server_label=server_labels[i],
                client_labels=client_labels,
            )
            all_clients_results.extend(
                gen_run(cmp_client_cmds, n=n, labels=labels, output_dir=output_dir)
            )

        # process results
        if not disable_csv:
            gen_export_csv(
                all_clients_results,
                output_dir=output_dir,
            )

        if not disable_table:
            cmp_export_table(
                all_clients_results,
                input_features,
                output_metrics,
                num_clients=num_clients,
                num_servers=num_servers,
                output_dir=output_dir,
                server_labels=server_labels,
            )

        if not disable_plot:
            logger.warning("Haven't supported plot for cmp yet")

    except Exception as e:
        raise RuntimeError(f"Process failed with error: {e}") from e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
