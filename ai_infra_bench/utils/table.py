"""Stdlib-only terminal table and summary metric formatting helpers."""

import json
import math
from typing import Any, List, Sequence


def print_table(title: str, rows: List[List[str]]) -> None:
    print()
    if not rows:
        return

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    title_line = f"| {title.center(len(border) - 4)} |"

    print(border)
    print(title_line)
    print(border)
    for idx, row in enumerate(rows):
        print(
            "| "
            + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
            + " |"
        )
        if idx == 0:
            print(border)
    print(border)
    print()


def fmt(value: Any, fmt: str = ".2f", suffix: str = "") -> str:
    """Format ``value`` with a printf-style format spec, e.g. ``fmt(1.23, ".2f")``."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value}{suffix}"
    if isinstance(value, float):
        return f"{value:{fmt}}{suffix}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def format_histogram_percentages(histogram: Sequence[int]) -> str:
    total = sum(histogram)
    if total == 0:
        return "[]"
    percentages = (f"{count / total:.2%}" for count in histogram)
    return f"[{', '.join(percentages)}]"


def mean_value(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return math.fsum(values) / len(values)


def percentile_value(values: Sequence[float | int], percentile: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * (percentile / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def format_mean(values: Sequence[float | int], precision: int = 2) -> str:
    avg = mean_value(values)
    if avg is None:
        return "N/A"
    return f"{avg:.{precision}f}"


def format_percentile(
    values: Sequence[float | int], percentile: float, precision: int = 2
) -> str:
    val = percentile_value(values, percentile)
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}"
