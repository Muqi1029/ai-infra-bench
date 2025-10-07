import logging
import os
from typing import Dict, List

from ai_infra_bench.utils import avg_std_strf

logger = logging.getLogger(__name__)


def export_md_table(data, input_features, metrics, labels, output_dir):
    table_path = os.path.join(output_dir, "table.md")

    logger.info(f"Writing table to {table_path}")
    md_tables_str = ""
    for label_idx, label in enumerate(labels):
        md_tables_str += f"Title: **{label}**\n"
        md_tables_str += (
            "| "
            + " | ".join(str(input_feature) for input_feature in input_features)
            + " |     | "
            + " | ".join(str(metric) for metric in metrics)
            + " |\n"
            + "| --- " * (len(input_features) + len(metrics) + 1)
            + "|\n"
        )
        for item_list in data[label_idx]:
            for input_feature in input_features:
                md_tables_str += "| " + f"{item_list[0][input_feature]:.2f}" + " "
            md_tables_str += "|     "
            for metric in metrics:
                md_tables_str += (
                    "| " + f"{avg_std_strf(metric, item_list, precision=2)}" + " "
                )
            md_tables_str += "|\n"
        md_tables_str += "\n" * 5

    with open(table_path, mode="w", encoding="utf-8") as f:
        f.write(md_tables_str)

    logger.info("Writing table DONE")


def export_csv(data, input_features, metrics, output_dir):
    csv_path = os.path.join(output_dir, "data.csv")

    logger.info(f"Writing CSV to {csv_path}")
    with open(csv_path, "w", encoding="utf-8") as f:
        # Write header
        f.write(",".join(input_features + metrics) + "\n")
        for item_list in data:
            for input_feature in input_features:
                f.write(f"{item_list[0][input_feature]:.2f},")
            for metric in metrics:
                f.write(f"{avg_std_strf(metric, item_list, precision=2)},")
            f.write("\n")

    logger.info("Writing CSV DONE")


def export_csv(data: List[Dict], output_dir):
    csv_path = os.path.join(output_dir, "full_data.csv")

    logger.info(f"Writing full csv file to {csv_path}")

    title = data[0][0].keys()
    title_len = len(title)

    with open(csv_path, "w", encoding="utf-8") as f:
        # headers
        f.write(",".join(title) + "\n")

        for item_list in data:
            # traverse each line
            for i, name in enumerate(title):
                f.write(avg_std_strf(name, item_list, sep="|", precision=4))
                if i != title_len - 1:
                    f.write(",")
            f.write("\n")
    print(f"Writing full csv file to {csv_path} DONE")
