import logging
import os
from typing import Dict, List

from ai_infra_bench.check import (
    check_client_labels,
    check_content_server_client_cmds,
    check_dir,
    check_param_in_cmd,
    check_server_labels,
    check_str_list_str,
    check_values_in_features_metrics,
)
from ai_infra_bench.modes.cmp import cmp_export_table
from ai_infra_bench.modes.gen import gen_export_csv, gen_run
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    kill_process_tree,
    maybe_create_labels,
    maybe_warmup,
    run_cmd,
    wait_for_server,
)

logger = logging.getLogger(__name__)


def cmp_bench(
    server_cmds: str | List[str],
    client_cmds: str | List[str],
    *,
    input_features: List[str],
    output_metrics: List[str],
    server_labels: None | List[str] = None,
    client_labels: None | List[str] = None,
    host="127.0.0.1",
    port=None,
    base_url: None | str = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
):

    server_cmds = check_str_list_str(server_cmds)
    client_cmds = check_str_list_str(client_cmds)

    check_content_server_client_cmds(server_cmds, [client_cmds])
    check_values_in_features_metrics(input_features, output_metrics)
    check_param_in_cmd("output-file", client_cmds)

    num_servers = len(server_cmds)
    num_clients = len(client_cmds)
    server_labels = check_server_labels(server_labels, num_servers)
    client_labels = check_client_labels(client_labels, [num_clients])[0]

    output_dir = check_dir(
        output_dir=output_dir, full_data_json_path=FULL_DATA_JSON_PATH
    )
    try:
        all_clients_results: List[List[Dict]] = []
        for server_idx in range(num_servers):
            server_cmd_str = server_cmds[server_idx]

            # launch server
            logger.info(f"RUNNING SERVER:\n{server_cmd_str}")
            server_process = run_cmd(server_cmd_str, is_block=False)

            logger.info("WAITING FOR THE SERVER TO BE LAUNCHED")
            wait_for_server(base_url=base_url or f"http://{host}:{port}", timeout=120)
            maybe_warmup(
                cmd=client_cmds[0],
                output_dir=output_dir,
                disable_warmup=disable_warmup,
            )
            labels = maybe_create_labels(
                num_clients=num_clients,
                server_label=server_labels[server_idx],
                client_labels=client_labels,
            )

            all_clients_results.extend(
                gen_run(client_cmds, n=n, labels=labels, output_dir=output_dir)
            )

            if server_process:
                server_process.terminate()

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
