import pytest

from ai_infra_bench.cli import main
from ai_infra_bench.version import __version__


@pytest.mark.parametrize("argv", [["-V"], ["--version"]])
def test_version_prints_version_and_exits(argv, capsys):
    with pytest.raises(SystemExit) as error:
        main(argv)

    assert error.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"
