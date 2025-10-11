import os
from typing import Dict, List

from ai_infra_bench.utils import TABLE_NAME, avg_std_strf, enter_decorate


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
