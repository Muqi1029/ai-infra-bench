import logging
import os
from typing import Callable, Dict, List, Tuple

from ai_infra_bench.modes.gen import gen_export_csv, gen_export_table, gen_plot

logger = logging.getLogger(__name__)

from ai_infra_bench.check import (
    check_client_labels,
    check_content_server_client_cmds,
    check_dir,
    check_param_in_cmd,
    check_server_labels,
    check_str_list_str,
)
from ai_infra_bench.modes.slo import slo_run
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    add_request_rate,
    kill_process_tree,
    maybe_create_labels,
    maybe_warmup,
    run_cmd,
    wait_for_server,
)


def slo_bench(
    server_cmds: str | List[str],
    client_cmds: str | List[str],
    *,
    input_features: List[str],
    output_metrics: List[str],
    request_rates: Tuple[int, int] | List[Tuple[int, int]],
    check_slo: Callable | List[Callable],
    server_labels: None | str | List[str] = None,
    client_labels: None | str | List[str] = None,
    host=None,
    port=None,
    base_url: None | str = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
) -> None:
    server_cmds = check_str_list_str(server_cmds)
    client_cmds = check_str_list_str(client_cmds)

    if isinstance(check_slo, Callable):
        check_slo = [check_slo] * len(client_cmds)
    if isinstance(request_rates, tuple):
        request_rates = [request_rates] * len(client_cmds)

    assert len(server_cmds) == len(client_cmds) == len(request_rates) == len(check_slo)

    check_content_server_client_cmds(server_cmds, [client_cmds])
    check_param_in_cmd("output-file", client_cmds)
    check_param_in_cmd("request-rate", client_cmds)
    check_param_in_cmd("max-concurrency", client_cmds)

    server_labels = check_server_labels(
        server_labels=server_labels, num_servers=len(server_cmds)
    )
    client_labels = check_client_labels(
        client_labels=client_labels, num_clients=[len(client_cmds)]
    )

    output_dir = check_dir(
        output_dir=output_dir, full_data_json_path=FULL_DATA_JSON_PATH
    )

    try:
        all_clients_results: List[List[Dict]] = []
        answers = []
        labels = maybe_create_labels(
            num_clients=len(client_cmds),
            server_label=server_labels[0],
            client_labels=client_labels[0],
        )
        for idx, (server_cmd, client_cmd, request_rate, check_slo_func) in enumerate(
            zip(server_cmds, client_cmds, request_rates, check_slo)
        ):
            # launch server
            logger.info(f"RUNNING SERVER:\n{server_cmd}")
            run_cmd(server_cmd, is_block=False)
            wait_for_server(base_url or f"http://{host}:{port}", 120)

            maybe_warmup(
                add_request_rate(client_cmds[idx], request_rate[0]),
                output_dir,
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
        raise RuntimeError(f"Process failed with error: {e}") from e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
