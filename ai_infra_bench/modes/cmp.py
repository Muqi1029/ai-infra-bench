import os
from typing import Dict, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_infra_bench.utils import (
    TABLE_NAME,
    avg_std_strf,
    colors,
    enter_decorate,
    graph_per_row,
)


@enter_decorate("PLOT TO HTML", filename="<input_feature>.html")
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


@enter_decorate("CMP EXPORT TBALE", filename=TABLE_NAME)
def cmp_export_table(
    all_clients_results: List[List[Dict]],
    input_features: List[str],
    output_metrics: List[Dict],
    num_clients: int,
    num_servers: int,
    output_dir: str,
    server_labels: List[str],
):
    if not all_clients_results or not all_clients_results[0]:
        raise ValueError("No data available to export.")

    if server_labels[0] is None:
        server_labels = [f"server_{i + 1}" for i in range(num_servers)]

    # header
    header_cells = input_features + [" - "]
    for output_metric in output_metrics:
        header_cells += [output_metric] + [" - "] * (len(server_labels) - 1)
    header_row = "| " + " | ".join(map(str, header_cells)) + " |"

    # sub header
    sub_header_cells = [" - "] * (len(input_features) + 1) + server_labels * len(
        output_metrics
    )
    sub_header_row = "| " + " | ".join(map(str, sub_header_cells)) + " |"

    separator_row = "| " + " | ".join(["---"] * len(header_cells)) + " |"
    lines = [header_row, sub_header_row, separator_row]

    for client_idx in range(num_clients):
        #
        row_values = []

        all_server_metrics = []
        for server_idx in range(num_servers):
            server_metrics = []
            idx = client_idx + server_idx * num_clients
            row_results = all_clients_results[idx]
            if server_idx == 0:
                for feature in input_features:
                    row_values.append(f"{row_results[0][feature]:.2f}")
                row_values.append("-")
            for metric in output_metrics:
                server_metrics.append(avg_std_strf(metric, row_results, precision=2))
            all_server_metrics.append(server_metrics)

        for i in range(len(output_metrics)):
            for j in range(num_servers):
                row_values.append(all_server_metrics[j][i])
        lines.append("| " + " | ".join(row_values) + " |")

    with open(os.path.join(output_dir, TABLE_NAME), mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))
