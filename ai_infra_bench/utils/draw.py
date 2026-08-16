import json
from enum import Enum, auto
from typing import List, Sequence

import numpy as np


class Color(Enum):
    LIGHT_CYAN = auto()
    LIGHT_GREEN = auto()
    LIGHT_YELLOW = auto()
    RED = auto()


def color_print(text: str, color: Color):
    RESET_CODE = "\033[0m"
    COLOR_TO_ANSI = {
        Color.LIGHT_CYAN: "\033[96m",
        Color.LIGHT_GREEN: "\033[92m",
        Color.LIGHT_YELLOW: "\033[93m",
        Color.RED: "\033[41m",
    }

    try:
        color_code = COLOR_TO_ANSI[color]
    except KeyError:
        raise NotImplementedError(f"{color} is not supported yet.")

    print(f"{color_code}{text}{RESET_CODE}", end="", flush=True)


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


def fmt(value, fmt: str = ".2f", suffix: str = "") -> str:
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


def format_mean(values: List[float], precision: int = 2) -> str:
    if not values:
        return "N/A"
    return f"{np.mean(values):.{precision}f}"


def format_percentile(
    values: List[float], percentile: float, precision: int = 2
) -> str:
    if not values:
        return "N/A"
    return f"{np.percentile(values, percentile):.{precision}f}"
