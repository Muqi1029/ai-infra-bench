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


@enter_decorate("CMP EXPORT TABLE", filename=TABLE_NAME)
def cmp_export_table(
    all_clients_results: List[List[Dict]],
    input_features: List[str],
    output_metrics: List[str],
    num_clients: int,
    num_servers: int,
    output_dir: str,
    server_labels: List[str],
):
    if not all_clients_results or not all_clients_results[0]:
        raise ValueError("No data available to export.")

    if server_labels is None or server_labels[0] is None:
        server_labels = [f"server_{i + 1}" for i in range(num_servers)]

    # --- 1. 动态构建表头 ---
    # 将 input_features 组合成标题，例如: "Config (input_len / output_len / rate)"
    config_header_name = f"Config ({' / '.join(input_features)})"

    header_cells = [config_header_name, "Metric"] + server_labels
    if num_servers == 2:
        header_cells.append("Diff (%)")

    header_row = "| " + " | ".join(header_cells) + " |"
    separator_row = "| " + " | ".join(["---"] * len(header_cells)) + " |"
    lines = [header_row, separator_row]

    # --- 2. 遍历每一个配置 (Client Config) ---
    for client_idx in range(num_clients):

        # 动态提取当前配置下所有 feature 的值
        # 索引逻辑: client_idx 对应第一个 server 的该配置结果
        first_server_res_list = all_clients_results[client_idx]
        first_sample = first_server_res_list[0]

        config_val_list = []
        for feat in input_features:
            val = first_sample.get(feat, "N/A")
            # 格式化数值：如果是浮点数保留两位，否则转字符串
            if isinstance(val, float):
                config_val_list.append(f"{val:.2f}")
            else:
                config_val_list.append(str(val))

        # 拼接后的配置字符串，例如 "1200.00 / 800.00 / 4.00"
        config_str = " / ".join(config_val_list)

        # --- 3. 遍历每一个指标 (Metric) ---
        for m_idx, metric in enumerate(output_metrics):
            row_values = []

            # 第一列：仅在指标块的第一行显示配置
            if m_idx == 0:
                row_values.append(f"**{config_str}**")
            else:
                row_values.append(" ")

            # 第二列：指标名称
            row_values.append(metric)

            # 后面几列：各个 Server 的数值
            numerical_means = []
            for s_idx in range(num_servers):
                idx = client_idx + s_idx * num_clients
                res_list = all_clients_results[idx]

                # 使用你原有的格式化函数获取 "均值 ± 标准差"
                display_str = avg_std_strf(metric, res_list, precision=2)
                row_values.append(display_str)

                # 为计算 Diff 提取纯数值均值
                try:
                    m_val = sum(r[metric] for r in res_list) / len(res_list)
                    numerical_means.append(m_val)
                except:
                    numerical_means.append(None)

            # 最后一列：动态计算两个 Server 间的差异
            if num_servers == 2:
                v1, v2 = numerical_means[0], numerical_means[1]
                if v1 is not None and v2 is not None and v1 != 0:
                    diff = (v2 - v1) / v1 * 100
                    row_values.append(f"{diff:+.2f}%")
                else:
                    row_values.append("-")

            lines.append("| " + " | ".join(row_values) + " |")

    # --- 4. 写入文件 ---
    output_path = os.path.join(output_dir, TABLE_NAME)
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# @enter_decorate("CMP EXPORT TBALE", filename=TABLE_NAME)
# def cmp_export_table(
#     all_clients_results: List[List[Dict]],
#     input_features: List[str],
#     output_metrics: List[Dict],
#     num_clients: int,
#     num_servers: int,
#     output_dir: str,
#     server_labels: List[str],
# ):
#     if not all_clients_results or not all_clients_results[0]:
#         raise ValueError("No data available to export.")
#
#     if server_labels[0] is None:
#         server_labels = [f"server_{i + 1}" for i in range(num_servers)]
#
#     # header
#     header_cells = input_features + [" - "]
#     for output_metric in output_metrics:
#         header_cells += [output_metric] + [" - "] * (len(server_labels) - 1)
#     header_row = "| " + " | ".join(map(str, header_cells)) + " |"
#
#     # sub header
#     sub_header_cells = [" - "] * (len(input_features) + 1) + server_labels * len(
#         output_metrics
#     )
#     sub_header_row = "| " + " | ".join(map(str, sub_header_cells)) + " |"
#
#     separator_row = "| " + " | ".join(["---"] * len(header_cells)) + " |"
#     lines = [header_row, sub_header_row, separator_row]
#
#     for client_idx in range(num_clients):
#         #
#         row_values = []
#
#         all_server_metrics = []
#         for server_idx in range(num_servers):
#             server_metrics = []
#             idx = client_idx + server_idx * num_clients
#             row_results = all_clients_results[idx]
#             if server_idx == 0:
#                 for feature in input_features:
#                     row_values.append(f"{row_results[0][feature]:.2f}")
#                 row_values.append("-")
#             for metric in output_metrics:
#                 server_metrics.append(avg_std_strf(metric, row_results, precision=2))
#             all_server_metrics.append(server_metrics)
#
#         for i in range(len(output_metrics)):
#             for j in range(num_servers):
#                 row_values.append(all_server_metrics[j][i])
#         lines.append("| " + " | ".join(row_values) + " |")
#
#     with open(os.path.join(output_dir, TABLE_NAME), mode="w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
