import os
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_infra_bench.check import (
    check_dir,
    check_input_features_metrics,
    check_output_file,
    check_param_in_cmd,
)
from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    add_request_rate,
    avg_std_strf,
    colors,
    graph_per_row,
    kill_process_tree,
    read_jsonl,
    run_cmd,
    sort_data_by_key,
    warmup,
)


def export_csv(data: List[Dict], output_dir):
    csv_path = os.path.join(output_dir, "full_data.csv")

    print(f"Writing full csv file to {csv_path}")

    title = data[0][0].keys()
    title_len = len(title)

    with open(csv_path, "w", encoding="utf-8") as f:
        # headers
        f.write(",".join(title) + "\n")

        for item_list in data:
            # traverse each line
            for i, name in enumerate(title):
                f.write(avg_std_strf(name, item_list, sep="| "))
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
    for item_list in data:
        for input_feature in input_features:
            md_tables_str += (
                "| " + avg_std_strf(input_feature, item_list, precision=2) + " "
            )
        md_tables_str += "|     "
        for metric in metrics:
            md_tables_str += "| " + avg_std_strf(metric, item_list, precision=2) + " "
        md_tables_str += "|\n"

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_tables_str)
    print("Writing table DONE")


def plot(data, input_features, metrics, label, output_dir):
    print("Ploting graphs in html")

    for input_feature in input_features:
        rows = (len(metrics) - 1) // graph_per_row + 1
        fig = make_subplots(rows=rows, cols=graph_per_row)

        x = [np.mean([item[input_feature] for item in item_list]) for item_list in data]
        cur_row, cur_col = 0, 0

        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[
                        np.mean([item[metric] for item in item_list])
                        for item_list in data
                    ],
                    name=f"{metric} (AVG)",
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


def client_slo(
    client_cmds: List[str],
    input_features: List[str],
    metrics: List[str],
    check_slo: Callable,
    request_rates: List[Tuple[int, int]],
    labels: List[str] = None,
    n=1,
    output_dir="output",
):
    if isinstance(client_cmds, str):
        client_cmds = [client_cmds]

    check_input_features_metrics(input_features, metrics)
    check_param_in_cmd("output-file", client_cmds)
    check_param_in_cmd("request-rate", client_cmds)
    check_param_in_cmd("max-concurrency", client_cmds)
    assert len(client_cmds) == len(request_rates)

    if not labels:
        labels = [
            datetime.now.strftime("%m%d") + f"_slo_exp_{i}"
            for i in range(len(client_cmds))
        ]
        print(
            f"The labels for this run is not set, it will be set {labels} by default respectively"
        )

    output_dir = check_dir(output_dir, FULL_DATA_JSON_PATH)

    try:
        # warmup
        print("Using the first client request for warm up")
        warmup(client_cmds[0], output_dir)

        data: List[Dict] = []
        for i in range(len(client_cmds)):
            print(f"\nRunning {i}-th client\n")
            left, right = request_rates[i]
            while left <= right:
                mid = (left + right) // 2
                cmd = add_request_rate(client_cmds[i], mid)

                inner_data = []
                for ii in range(n):
                    output_file = f"{labels[i]}_client_{i:02d}_{ii:02d}.jsonl"
                    output_file = os.path.join(
                        output_dir, FULL_DATA_JSON_PATH, output_file
                    )
                    cmd += f" --output-file {output_file}"
                    run_cmd(cmd, is_block=True)
                    inner_data.append(read_jsonl(output_file)[-1])

                union_avg_item = {}
                for key in inner_data[0].keys():
                    if isinstance(inner_data[0][key], str):
                        union_avg_item[key] = inner_data[0][key]
                    else:
                        union_avg_item[key] = np.mean(item[key] for item in inner_data)
                if check_slo(union_avg_item):
                    left = mid + 1
                else:
                    right = mid - 1
                data.append(inner_data)
        # sort data in request_rate
        sorted_data = sort_data_by_key("max-concurrency", data)

        export_table(
            data=sorted_data,
            input_features=input_features,
            metrics=metrics,
            label=labels,
            output_dir=output_dir,
        )
        plot(
            data=sorted_data,
            input_features=input_features,
            metrics=metrics,
            label=labels,
            output_dir=output_dir,
        )
        export_csv(sorted_data, output_dir)

    except Exception as e:
        print(e)
        kill_process_tree(os.getpid(), include_parent=False)


def client_gen(
    client_cmds: Union[List[str], str],
    input_features,
    metrics,
    label=None,
    n=1,
    output_dir="output",
):
    if isinstance(client_cmds, str):
        client_cmds = [client_cmds]

    check_input_features_metrics(input_features, metrics)
    check_output_file(client_cmds)

    if not label:
        label = datetime.now.strftime("%m%d") + "_slo_exp"
        print(
            f"The label for this server is not set, it will be set {label} by default"
        )

    output_dir = check_dir(output_dir, FULL_DATA_JSON_PATH)
    print(f"{output_dir=}")

    try:
        # warmup
        print("Using the first client request for warm up")
        warmup(client_cmds[0], output_dir)

        data: List[Dict] = []
        for i, cmd in enumerate(client_cmds):
            print(f"\nRunning {i}-th client\n")

            inner_data = []
            for ii in range(n):
                output_file = f"client_{i:02d}_{ii:02d}.jsonl"
                output_file = os.path.join(output_dir, FULL_DATA_JSON_PATH, output_file)
                cmd += f" --output-file {output_file}"
                run_cmd(cmd, is_block=True)
                inner_data.append(read_jsonl(output_file)[-1])

            data.append(inner_data)

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

    except Exception as e:
        print(e)
        kill_process_tree(os.getpid(), include_parent=False)
