import os
import time
from typing import Dict, List

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


def general_export_table(data, input_features, metrics, labels, output_dir):
    print(f"Writing table to {os.path.join(output_dir, 'table.md')}")
    md_tables_str = ""
    for label_idx, label in enumerate(labels):

        # add title
        md_tables_str += f"Title: **{label}**\n"
        md_tables_str += (
            "| "
            + " | ".join(str(input_feature) for input_feature in input_features)
            + " |     | "
            + " | ".join(str(metric) for metric in metrics)
            + " |\n"
        )

        # add delimiter
        md_tables_str += "| --- " * (len(input_features) + len(metrics) + 1) + "|\n"

        # add data
        for item in data[label_idx]:

            # FIXME(muqi1029): only support float or int features and metrics
            for input_feature in input_features:
                md_tables_str += "| " + f"{item[input_feature]:.2f}" + " "
            md_tables_str += "|     "
            for metric in metrics:
                md_tables_str += "| " + f"{item[metric]:.2f}" + " "
            md_tables_str += "|\n"

        md_tables_str += "\n" * 5
    with open(os.path.join(output_dir, "table.md"), mode="w", encoding="utf-8") as f:
        f.write(md_tables_str)
    print("Writing table DONE")


def general_plot(data, input_features, metrics, labels, output_dir):
    print("Ploting graphs in html")
    for i, label in enumerate(labels):
        for input_feature in input_features:

            rows = (len(metrics) - 1) // graph_per_row + 1
            # fig = make_subplots(rows=rows, cols=graph_per_row, subplot_titles=metrics)
            fig = make_subplots(rows=rows, cols=graph_per_row)

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
    print("Ploting graphs DONE")


def general_bench(
    server_cmds,
    client_cmds,
    *,
    input_features,
    metrics,
    labels,
    host,
    port,
    output_dir="output",
):
    check_server_client_cmds(server_cmds, client_cmds, labels=labels)

    os.makedirs(output_dir, exist_ok=False)

    pbar = tqdm(enumerate(zip(server_cmds, client_cmds)))

    data: List[List[Dict]] = []

    try:
        for server_idx, (server_cmd, client_cmd) in pbar:

            pbar.set_description(f"======= Running {server_idx + 1}-th server =======")

            # launch server
            server_process = run_cmd(server_cmd, is_block=False)

            wait_for_server(base_url=f"http://{host}:{port}", timeout=120)

            # warmup
            print("Begin Warmup")
            warmup(client_cmd[0], output_dir)
            print("Warmup DONE")

            inner_data: List[Dict] = []

            # launch client
            for client_idx, cmd in enumerate(client_cmd):
                output_file = dummy_get_filename(client_idx, label=labels[server_idx])
                output_file = os.path.join(output_dir, output_file)
                cmd += f" --output-file {output_file}"
                run_cmd(cmd, is_block=True)

                inner_data.append(read_jsonl(output_file)[-1])

                time.sleep(5)

            data.append(inner_data)

            server_process.terminate()

            time.sleep(5)  # wait it to exit gracefully and completely

            pbar.update(1)

        pbar.close()

        general_export_table(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
        general_plot(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
