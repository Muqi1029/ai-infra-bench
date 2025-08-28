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


def cmp_plot(data, input_features, metrics, labels, output_dir):
    print("Ploting graphs in html")

    cur_row, cur_col = 0, 0
    num_client_settings = len(data[0])
    num_server_settings = len(data)

    # there are totally len(input_features) html files
    for input_feature in input_features:
        rows = (len(metrics) - 1) // graph_per_row + 1
        cols = graph_per_row
        fig = make_subplots(rows=rows, cols=cols)

        # there totally are len(metric) subplots
        for metric in metrics:

            # each server is a line
            for server_idx in range(num_server_settings):

                fig.add_trace(
                    go.Scatter(
                        x=[
                            data[server_idx][i][input_feature]
                            for i in range(num_client_settings)
                        ],
                        y=[
                            data[server_idx][i][metric]
                            for i in range(num_client_settings)
                        ],
                        name=labels[server_idx],
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(
                            color=colors[server_idx % len(colors)],
                            width=3,
                        ),
                        hovertemplate=f"<br>{input_feature}: %{{x}}<br>{metric}: %{{y}}<br><extra></extra>",
                    ),
                    row=cur_row + 1,
                    col=cur_col + 1,
                )
            fig.update_xaxes(title_text=input_feature, row=cur_row + 1, col=cur_col + 1)
            fig.update_yaxes(title_text=metric, row=cur_row + 1, col=cur_col + 1)

            # one subplot is over
            cur_col += 1
            if cur_col == graph_per_row:
                cur_col = 0
                cur_row += 1

        fig.update_layout(title_text="_vs_".join(labels) + "_in_" + input_feature)
        html_name = f"{input_feature}_" + "_vs_".join(labels) + ".html"
        fig.write_html(os.path.join(output_dir, html_name))

    print("Ploting graphs DONE")


def cmp_export_table(data, input_features, metrics, labels, output_dir):
    print(f'Writing table to {os.path.join(output_dir, "table.md")}')
    md_tables_str = ""
    common_title = (
        "| "
        + " | ".join([str(input_feature) for input_feature in input_features])
        + " |     | "
        + " | ".join([str(label) for label in labels])
        + " |\n"
        + "| --- " * (len(input_features) + len(labels) + 1)
        + "|\n"
    )

    for metric in metrics:
        md_tables_str += f"Metric: **{metric}**\n" + common_title

        # each client setting is a line
        for client_idx in range(len(data[0])):
            # each label setting is a column
            for label_idx in range(len(labels)):

                item = data[label_idx][client_idx]
                if label_idx == 0:
                    # only read the first label's client setting since they are the same
                    for input_feature in input_features:
                        md_tables_str += "| " + f"{item[input_feature]:.2f}" + " "
                    md_tables_str += "|     "
                md_tables_str += "| " + f"{item[metric]:.2f}" + " "
            md_tables_str += "|\n"
        md_tables_str += "\n" * 5

    with open(os.path.join(output_dir, "table.md"), "w", encoding="utf-8") as f:
        f.write(md_tables_str)
    print("Writing table DONE")


def cmp_bench(
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
    try:
        check_server_client_cmds(server_cmds, client_cmds, labels=labels)
        os.makedirs(output_dir, exist_ok=False)

        data: List[List[Dict]] = []
        pbar = tqdm(enumerate(server_cmds))
        for server_idx, server_cmd in pbar:
            pbar.set_description(f"======= Running {server_idx + 1}-th server =======")

            # launch server
            server_process = run_cmd(server_cmd, is_block=False)

            wait_for_server(base_url=f"http://{host}:{port}", timeout=120)

            # warmup
            print("Begin Warmup")
            warmup(client_cmds[0], output_dir)
            print("Warmup over")

            # launch_client
            inner_data: List[Dict] = []
            for client_idx, client_cmd in enumerate(client_cmds):
                output_file = dummy_get_filename(client_idx, label=labels[server_idx])
                output_file = os.path.join(output_dir, output_file)
                client_cmd += f" --output-file {output_file}"
                run_cmd(client_cmd, is_block=True)
                inner_data.append(read_jsonl(output_file)[-1])

                time.sleep(5)

            data.append(inner_data)

            server_process.terminate()

            time.sleep(5)  # wait it to exit gracefully and completely

            pbar.update(1)

        pbar.close()

        cmp_export_table(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
        cmp_plot(
            data=data,
            input_features=input_features,
            metrics=metrics,
            labels=labels,
            output_dir=output_dir,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
