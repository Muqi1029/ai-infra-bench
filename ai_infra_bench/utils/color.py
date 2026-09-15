import logging
import os
import sys
from enum import Enum, auto


class Color(Enum):
    LIGHT_CYAN = auto()
    LIGHT_GREEN = auto()
    LIGHT_YELLOW = auto()
    RED = auto()


RESET_CODE = "\033[0m"
COLOR_TO_ANSI = {
    Color.LIGHT_CYAN: "\033[96m",
    Color.LIGHT_GREEN: "\033[92m",
    Color.LIGHT_YELLOW: "\033[93m",
    Color.RED: "\033[41m",
}
LEVEL_TO_COLOR = {
    logging.DEBUG: Color.LIGHT_CYAN,
    logging.INFO: Color.LIGHT_GREEN,
    logging.WARNING: Color.LIGHT_YELLOW,
    logging.ERROR: Color.RED,
    logging.CRITICAL: Color.RED,
}


def colorize(text: str, color: Color) -> str:
    try:
        color_code = COLOR_TO_ANSI[color]
    except KeyError:
        raise NotImplementedError(f"{color} is not supported yet.")
    return f"{color_code}{text}{RESET_CODE}"


def use_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stderr if stream is None else stream
    return hasattr(stream, "isatty") and stream.isatty()


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        color = LEVEL_TO_COLOR.get(record.levelno)
        if color is None:
            return formatted
        return colorize(formatted, color)


def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(levelname)s(%(asctime)s):  %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    handler = logging.StreamHandler()
    formatter_cls = ColoredFormatter if use_color() else logging.Formatter
    handler.setFormatter(formatter_cls(fmt, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler])
