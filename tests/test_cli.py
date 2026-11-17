import subprocess
import sys

import pytest

from ai_infra_bench.cli import _COMMANDS, main
from ai_infra_bench.version import __version__


@pytest.mark.parametrize("argv", [["-V"], ["--version"]])
def test_version_prints_version_and_exits(argv, capsys):
    with pytest.raises(SystemExit) as error:
        main(argv)

    assert error.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"


@pytest.mark.parametrize("subcommand", sorted(_COMMANDS))
def test_subcommand_help_is_lazy_and_fast(subcommand):
    script = """
import sys
BLOCKED = {
    'torch', 'transformers', 'datasets', 'safetensors',
    'huggingface_hub', 'aiohttp', 'numpy', 'omegaconf', 'requests',
}
class BlockHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
from ai_infra_bench.cli import main
try:
    main([sys.argv[1], '--help'])
except SystemExit as exc:
    assert exc.code == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script, subcommand],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
