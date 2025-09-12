from typing import Dict, List

from ai_infra_bench.client import export_csv, export_table, plot


def export_csv_table_html(
    data: List[Dict], input_features, metrics, label, output_dir="."
):
    export_table(data, input_features, metrics, label, output_dir)
    export_csv(data, output_dir)
    plot(data, input_features, metrics, label, output_dir)
