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
)
from ai_infra_bench.modes.gen import gen_export_csv, gen_export_table, gen_plot, gen_run
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    kill_process_tree,
    maybe_create_labels,
    maybe_warmup,
    run_cmd,
    wait_for_server,
)

logger = logging.getLogger(__name__)


def gen_bench(
    server_cmds: str | List[str],
    client_cmds: str | List[str] | List[List[str]],
    *,
    input_features: List[str],
    output_metrics: List[str],
    server_labels: None | str | List[str] = None,
    client_labels: None | str | List[str] | List[List[str]] = None,
    host=None,
    port=None,
    base_url: None | str = None,
    n: int = 1,
    output_dir: str = "output",
    disable_warmup: bool = False,
    disable_plot: bool = False,
    disable_table: bool = False,
    disable_csv: bool = False,
):
    # check server_cmds type
    server_cmds = check_str_list_str(server_cmds)

    # check client_cmds type
    if isinstance(client_cmds, str):
        client_cmds = [[client_cmds]]
    elif isinstance(client_cmds, list):
        if all(isinstance(cmd, str) for cmd in client_cmds):
            # convert from List[str] → List[List[str]]
            client_cmds = [client_cmds]
        elif all(
            isinstance(cmds, list) and all(isinstance(c, str) for c in cmds)
            for cmds in client_cmds
        ):
            # already List[List[str]]
            pass
        else:
            raise ValueError(
                f"client_cmds must be List[str] or List[List[str]], got {client_cmds!r}"
            )
    else:
        raise ValueError(f"client_cmds must be str or List, got {type(client_cmds)}")

    # check content
    check_content_server_client_cmds(server_cmds, client_cmds)

    assert len(server_cmds) == len(client_cmds)

    # check output-file
    for client_cmd_list in client_cmds:
        check_param_in_cmd("output-file", client_cmd_list)

    # check server_labels
    server_labels = check_server_labels(
        server_labels=server_labels, num_servers=len(server_cmds)
    )

    # check client_labels
    client_labels = check_client_labels(
        client_labels=client_labels,
        num_clients=[len(client_cmd_list) for client_cmd_list in client_cmds],
    )
    assert len(client_labels) == len(client_cmds)

    output_dir = check_dir(
        output_dir=output_dir, full_data_json_path=FULL_DATA_JSON_PATH
    )

    try:
        all_clients_results: List[List[Dict]] = []
        for idx, (server_cmd_str, client_cmd_list) in enumerate(
            zip(server_cmds, client_cmds)
        ):
            # launch server
            logger.info(f"RUNNING SERVER:\n{server_cmd_str}")
            server_process = run_cmd(server_cmd_str, is_block=False)

            logger.info("WAITING FOR THE SERVER TO BE LAUNCHED")
            wait_for_server(base_url=base_url or f"http://{host}:{port}", timeout=120)

            maybe_warmup(
                cmd=client_cmd_list[0],
                output_dir=output_dir,
                disable_warmup=disable_warmup,
            )

            all_clients_results.extend(
                gen_run(
                    client_cmds=client_cmd_list,
                    n=n,
                    labels=maybe_create_labels(
                        num_clients=len(client_cmd_list),
                        server_label=server_labels[idx],
                        client_labels=client_labels[idx],
                    ),
                    output_dir=output_dir,
                )
            )
            if server_process:
                server_process.terminate()

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
