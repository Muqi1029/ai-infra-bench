import json
import types

import pytest

from ai_infra_bench import cli
from ai_infra_bench import token as token_mod
from ai_infra_bench.cli import main as cli_main


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(item) for item in ids)


def test_cli_help_does_not_import_subcommands(monkeypatch):
    def fail_import(name, package=None):
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(cli.importlib, "import_module", fail_import)
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0


def test_cli_dispatches_token(monkeypatch):
    imported = []

    def fake_import(name, package=None):
        imported.append(name)
        module = types.ModuleType(name)
        module.main = lambda argv: 0
        return module

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    assert cli_main(["token", "--tokenizer", "dummy", "hello"]) == 0
    assert imported == ["ai_infra_bench.token"]


def test_encode_prints_ids_and_count(monkeypatch, capsys):
    monkeypatch.setattr(token_mod, "load_tokenizer", lambda _: FakeTokenizer())

    assert token_mod.main(["--tokenizer", "dummy", "ab"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "text": "ab",
        "ids": [97, 98],
        "count": 2,
    }


def test_decode_prints_text(monkeypatch, capsys):
    monkeypatch.setattr(token_mod, "load_tokenizer", lambda _: FakeTokenizer())

    assert token_mod.main(["--tokenizer", "dummy", "--decode", "[97, 98]"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "text": "ab",
        "ids": [97, 98],
        "count": 2,
    }


def test_decode_reads_input_ids_with_nested_shape(monkeypatch, capsys):
    monkeypatch.setattr(token_mod, "load_tokenizer", lambda _: FakeTokenizer())
    payload = json.dumps({"input_ids": [[[97], [98]], [[99, 100]]]})

    assert token_mod.main(["--tokenizer", "dummy", "--decode", payload]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "text": [["a", "b"], ["cd"]],
        "ids": [[[97], [98]], [[99, 100]]],
        "count": 4,
    }


def test_decode_reads_input_ids_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(token_mod, "load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(
        token_mod.sys,
        "stdin",
        types.SimpleNamespace(read=lambda: '{"input_ids": [[97, 98], [99]]}\n'),
    )

    assert token_mod.main(["--tokenizer", "dummy", "--decode", "--from-stdin"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "text": ["ab", "c"],
        "ids": [[97, 98], [99]],
        "count": 3,
    }


def test_encode_reads_text_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr(token_mod, "load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(
        token_mod.sys, "stdin", types.SimpleNamespace(read=lambda: "ab\n")
    )

    assert token_mod.main(["--tokenizer", "dummy", "--from-stdin"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "text": "ab",
        "ids": [97, 98],
        "count": 2,
    }


def test_from_stdin_rejects_a_positional_argument():
    with pytest.raises(SystemExit):
        token_mod.main(["--tokenizer", "dummy", "--from-stdin", "ab"])
