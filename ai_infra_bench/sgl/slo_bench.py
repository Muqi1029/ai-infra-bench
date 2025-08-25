import os
from typing import Callable, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ai_infra_bench.utils import (
    check_server_client_cmds,
    colors,
    dummy_get_filename,
    graph_per_row,
    kill_process_tree,
    read_jsonl,
    run_cmd,
    wait_for_server,
    warmup,
)


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


def add_request_rate(cmd: str, rate: int):
    cmd += f" --max-concurrency {rate} --request-rate {rate}"
    if "num-prompt" not in cmd:
        cmd += f" --num-prompt {rate * 10}"
    return cmd


def slo_export_tables(
    data: List[List[Dict]],
    input_features: List[str],
    metrics: List[str],
    labels: List[str],
    output_dir: str,
):
    print(f"Writing table to {os.path.join(output_dir, 'table.md')}")
    md_tables_str = ""
    for server_data, label in zip(data, labels):
        server_data.sort(
            key=lambda item: [item[input_feature] for input_feature in input_features]
        )
        md_tables_str += f"Title: **{label}**\n"
        md_tables_str += (
            "| "
            + " | ".join(str(input_feature) for input_feature in input_features)
            + " |     | "
            + " | ".join(str(metric) for metric in metrics)
            + " |\n"
        )
        for item in server_data:
            for input_feature in input_features:
                md_tables_str += "| " + f"{item[input_feature]:.2f}" + " "
            md_tables_str += "|     "
            for metric in metrics:
                md_tables_str += "| " + f"{item[metric]:.2f}" + " "
            md_tables_str += "|\n"
        md_tables_str += "\n" * 5
    with open(os.path.join(output_dir, "table.md"), mode="w", encoding="utf-8") as f:
        f.write(md_tables_str)
    print("Writing table done")


def slo_plot(
    data: List[List[Dict]],
    input_features: List[str],
    metrics: List[str],
    labels: List[str],
    output_dir: str,
):
    print("Ploting graphs in html")
    for i, label in enumerate(labels):
        for input_feature in input_features:
            rows = (len(metrics) - 1) // graph_per_row + 1
            cols = graph_per_row

            fig = make_subplots(rows=rows, cols=cols)

            x = [item[input_feature] for item in data[i]]
            cur_row, cur_col = 0, 0
            for metric in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=[item[metric] for item in data[i]],
                        name=f"{label}/{metric}",
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(
                            color=colors[
                                (cur_row * graph_per_row + cur_col) % len(colors)
                            ],
                            width=3,
                        ),
                        hovertemplate=f"<br>{input_feature}: %{{x}}<br>{metric}: %{{y}}<br><extra></extra>",
                    ),
                    row=cur_row + 1,
                    col=cur_col + 1,
                )
                fig.update_xaxes(
                    title_text=input_feature, row=cur_row + 1, col=cur_col + 1
                )
                fig.update_yaxes(title_text=metric, row=cur_row + 1, col=cur_col + 1)
                cur_col += 1
                if cur_col == graph_per_row:
                    cur_row += 1
                    cur_col = 0
            fig.update_layout(title_text=label)
            fig.write_html(os.path.join(output_dir, f"{label}_{input_feature}.html"))

    print("Plotting graphs DONE")


def slo_bench(
    server_cmds: List[str],
    client_cmds: List[str],
    *,
    request_rates: List[Tuple[int, int]],
    input_features: List[str],
    metrics: List[str],
    labels: List[str],
    host,
    port,
    check_slo: Callable[[Dict], bool],
    output_dir: str = "output",
):
    try:
        slo_check_params(server_cmds, client_cmds, labels)
        os.makedirs(output_dir, exist_ok=False)

        data: List[List[Dict]] = []

        for idx, server_cmd in tqdm(enumerate(server_cmds)):
            # launch server
            run_cmd(server_cmd, is_block=False)
            wait_for_server(f"http://{host}:{port}", 120)

            left, right = request_rates[idx]

            warmup_cmd = add_request_rate(client_cmds[idx], left)
            warmup(warmup_cmd, output_dir)

            inner_data: List[Dict] = []
            client_idx = 0
            while left <= right:
                mid = (left + right) // 2

                cmd = add_request_rate(client_cmds[idx], mid)
                output_file = os.path.join(
                    output_dir, dummy_get_filename(client_idx, label=labels[idx])
                )
                client_idx += 1
                cmd += f" --output-file {output_file}"

                print(f"==== Running {mid} ====")
                run_cmd(cmd, is_block=True)

                item = read_jsonl(output_file)[-1]
                if check_slo(item):
                    left = mid + 1
                else:
                    right = mid - 1
                inner_data.append(item)

            print(f"\033[92m The maximum concurrency satisfying SLO is {right} \033[0m")
            data.append(inner_data)
        slo_export_tables(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
        slo_plot(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
