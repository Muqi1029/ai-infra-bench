"""Version for ai-infra-bench-dataset.

Build / editable monorepo: read ai_infra_bench/version.py (single source of truth).
Installed wheel: fall back to package metadata.
"""

from pathlib import Path

_version_file = Path(__file__).resolve().parents[2] / "ai_infra_bench" / "version.py"

if _version_file.is_file():
    _ns = {}
    exec(_version_file.read_text(encoding="utf-8"), _ns)
    __version__ = _ns["__version__"]
else:
    from importlib.metadata import version

    __version__ = version("ai-infra-bench-dataset")
