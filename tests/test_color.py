import logging
from io import StringIO

from ai_infra_bench.utils.color import (
    Color,
    ColoredFormatter,
    color_print,
    colorize,
    use_color,
)


def test_colorize():
    text = "hello"
    cyan = colorize(text, Color.LIGHT_CYAN)
    assert cyan == "\033[96mhello\033[0m"

    green = colorize(text, Color.LIGHT_GREEN)
    assert green == "\033[92mhello\033[0m"

    yellow = colorize(text, Color.LIGHT_YELLOW)
    assert yellow == "\033[93mhello\033[0m"

    red = colorize(text, Color.RED)
    assert red == "\033[41mhello\033[0m"


def test_color_print(capsys):
    color_print("test text", Color.LIGHT_CYAN)
    captured = capsys.readouterr()
    assert captured.out == "\033[96mtest text\033[0m"


def test_colored_formatter():
    formatter = ColoredFormatter("%(levelname)s: %(message)s")
    record_warning = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="warn msg",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record_warning)
    assert "\033[93mWARNING: warn msg\033[0m" == formatted

    record_info = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="info msg",
        args=(),
        exc_info=None,
    )
    formatted_info = formatter.format(record_info)
    assert formatted_info == "INFO: info msg"


def test_use_color_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert use_color() is False
