import logging

from ai_infra_bench.version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s(%(asctime)s):  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = ["__version__"]
