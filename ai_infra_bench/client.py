import os
from datetime import datetime
from typing import Dict, List, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ai_infra_bench.utils import (
    check_dir,
    check_input_features_metrics,
    colors,
    graph_per_row,
    kill_process_tree,
    read_jsonl,
    run_cmd,
    warmup,
)


def export_csv(data: List[Dict], output_dir):
    csv_path = os.path.join(output_dir, "full_data.csv")

    print(f"Writing full csv file to {csv_path}")

    title = data[0].keys()
    title_len = len(title)

    with open(csv_path, "w", encoding="utf-8") as f:
        # headers
        f.write(",".join(title) + "\n")

        for item in data:
            # traverse each line
            for i, name in enumerate(title):
                f.write(str(item[name]))
                if i != title_len - 1:
                    f.write(",")
            f.write("\n")
    print(f"Writing full csv file to {csv_path} DONE")


def export_table(data, input_features, metrics, label, output_dir):
    table_path = os.path.join(output_dir, "table.md")

    print(f"Writing table to {table_path}")
    md_tables_str = f"Title: **{label}**\n"
    md_tables_str += (
        "| "
        + " | ".join(str(input_feature) for input_feature in input_features)
        + " |     | "
        + " | ".join(str(metric) for metric in metrics)
        + " |\n"
        + "| --- " * (len(input_features) + len(metrics) + 1)
        + "|\n"
    )
    for item in data:
        for input_feature in input_features:
            md_tables_str += "| " + f"{item[input_feature]:.2f}" + " "
        md_tables_str += "|     "
        for metric in metrics:
            md_tables_str += "| " + f"{item[metric]:.2f}" + " "
        md_tables_str += "|\n"

    with open(table_path) as f:
        f.write(md_tables_str)
    print("Writing table DONE")


def plot(data, input_features, metrics, label, output_dir):
    print("Ploting graphs in html")
    for input_feature in input_features:

        rows = (len(metrics) - 1) // graph_per_row + 1
        # fig = make_subplots(rows=rows, cols=graph_per_row, subplot_titles=metrics)
        fig = make_subplots(rows=rows, cols=graph_per_row)

        x = [item[input_feature] for item in data]
        cur_row, cur_col = 0, 0

        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[item[metric] for item in data],
                    name=f"{metric}",
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(
                        color=colors[(cur_row * graph_per_row + cur_col) % len(colors)],
                        width=3,
                    ),
                    hovertemplate=f"<br>{input_feature}: %{{x}}<br>{metric}: %{{y}}<br><extra></extra>",
                ),
                row=cur_row + 1,
                col=cur_col + 1,
            )
            fig.update_xaxes(title_text=input_feature, row=cur_row + 1, col=cur_col + 1)
            fig.update_yaxes(title_text=metric, row=cur_row + 1, col=cur_col + 1)

            cur_col += 1
            if cur_col == graph_per_row:
                cur_row += 1
                cur_col = 0
        fig.update_layout(title_text="")
        fig.write_html(os.path.join(output_dir, f"{label}_{input_feature}.html"))
    print("Ploting graphs DONE")


def run_client(
    client_cmds: Union[List[str], str],
    input_features,
    metrics,
    label=None,
    output_dir="output",
):
    check_input_features_metrics(input_features, metrics)

    if not label:
        label = datetime.now.strftime("%m%d") + "_exp"
    full_data_json_path = "full_data_json"
    check_dir(output_dir, full_data_json_path)

    try:
        # warmup
        warmup(client_cmds[0], output_dir)

        data: List[Dict] = []
        for i, cmd in tqdm(enumerate(client_cmds), desc="Running client cmds"):
            output_file = f"client_{i:02d}.jsonl"
            output_file = os.path.join(output_dir, full_data_json_path, output_file)
            cmd += f" --output-file {output_file}"
            run_cmd(cmd, is_block=True)
            data.append(read_jsonl(output_file)[-1])

        export_table(
            data=data,
            input_features=input_features,
            metrics=metrics,
            label=label,
            output_dir=output_dir,
        )
        plot(
            data=data,
            input_features=input_features,
            metrics=metrics,
            label=label,
            output_dir=output_dir,
        )
        export_csv(data, output_dir)

    except Exception:
        kill_process_tree(os.getpid(), include_parent=False)
