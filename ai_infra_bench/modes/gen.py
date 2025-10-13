import csv
import logging
import math
import os
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ai_infra_bench.utils import (
    CSV_NAME,
    FULL_DATA_JSON_PATH,
    TABLE_NAME,
    avg_std_strf,
    colors,
    enter_decorate,
    graph_per_row,
    read_jsonl,
    run_cmd,
)

logger = logging.getLogger(__name__)


@enter_decorate(title="GEN RUN")
def gen_run(
    client_cmds: List[str], n: int, labels: List[str], output_dir: str
) -> List[List[Dict]]:
    all_clients_results: List[List[Dict]] = []
    for client_id, base_cmd in tqdm(
        enumerate(client_cmds), desc="Running clients", position=0
    ):
        client_results = []
        label = labels[client_id]
        for run_id in tqdm(
            range(n), desc=f"Client {client_id:02d} runs", position=1, leave=False
        ):
            output_filename = (
                f"{label}_run{run_id:02d}.jsonl" if n != 1 else f"{label}.jsonl"
            )

            output_path = os.path.join(
                output_dir,
                FULL_DATA_JSON_PATH,
                output_filename,
            )

            run_command = f"{base_cmd} --output-file {output_path}"

            run_cmd(run_command, is_block=True)

            # Read only the last JSON entry (most recent result)
            last_record = read_jsonl(output_path)[-1]
            client_results.append(last_record)

        all_clients_results.append(client_results)

    return all_clients_results


@enter_decorate("EXPORT_TABLE", filename=TABLE_NAME)
def gen_export_table(
    all_clients_results: List[List[Dict[str, Any]]],
    input_features: List[str],
    output_metrics: List[str],
    output_dir: str,
    server_label: str = "Results",
):
    # title
    header_cells = input_features + [" - "] + output_metrics
    header_row = "| " + " | ".join(map(str, header_cells)) + " |"
    separator_row = "| " + " | ".join(["---"] * len(header_cells)) + " |"

    lines = [f"### {server_label}", header_row, separator_row]

    for row_results in all_clients_results:
        row_values = []
        for feature in input_features:
            row_values.append(f"{row_results[0][feature]:.2f}")
        row_values.append("-")
        for metric in output_metrics:
            row_values.append(
                avg_std_strf(key=metric, item_list=row_results, precision=2)
            )
        lines.append("| " + " | ".join(row_values) + " |")

    with open(os.path.join(output_dir, TABLE_NAME), mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@enter_decorate("PLOT TO HTML", filename="<input_feature>.html")
def gen_plot(
    all_clients_results: List[List[Dict[str, Any]]],
    input_features: List[str],
    output_metrics: List[str],
    output_dir: str,
    server_label: str | None = None,
):
    for feature in input_features:
        num_graphs = len(output_metrics)
        num_rows = math.ceil(num_graphs / graph_per_row)

        fig = make_subplots(rows=num_rows, cols=graph_per_row)
        x_values = [
            np.mean([item[feature] for item in client])
            for client in all_clients_results
        ]

        for idx, metric in enumerate(output_metrics):
            row, col = divmod(idx, graph_per_row)

            y_values = [
                np.mean([item[metric] for item in client])
                for client in all_clients_results
            ]
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    name=f"{metric} (AVG)",
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color=color, width=3),
                    hovertemplate=f"<br>{feature}: %{{x}}<br>{metric}: %{{y}}<extra></extra>",
                ),
                row=row + 1,
                col=col + 1,
            )

            fig.update_xaxes(title_text=feature, row=row + 1, col=col + 1)
            fig.update_yaxes(title_text=metric, row=row + 1, col=col + 1)

        fig.update_layout(
            title_text=f"{server_label} - {feature}" if server_label else feature,
            showlegend=True,
            height=300 * num_rows,
            width=400 * graph_per_row,
            margin=dict(t=50, b=30, l=30, r=30),
        )

        output_path = os.path.join(
            output_dir,
            (f"{server_label}_{feature}.html" if server_label else f"{feature}.html"),
        )
        fig.write_html(output_path)


@enter_decorate("EXPORT_CSV", filename=f"<server_label>_{CSV_NAME}")
def gen_export_csv(
    all_clients_results: List[List[Dict]],
    output_dir: str,
    server_label: str | None = None,
    sep: str = "|",
    precision: int = 2,
):
    if not all_clients_results or not all_clients_results[0]:
        raise ValueError("No data available to export.")

    headers = list(all_clients_results[0][0].keys())

    with open(
        os.path.join(
            output_dir, f"{server_label}_{CSV_NAME}" if server_label else CSV_NAME
        ),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for client_results in all_clients_results:
            row = [
                avg_std_strf(metric, client_results, sep=sep, precision=precision)
                for metric in headers
            ]
            writer.writerow(row)
