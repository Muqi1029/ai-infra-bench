import logging

from ai_infra_bench.client import client_cmp, client_gen, client_slo
from ai_infra_bench.version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s(%(asctime)s):  %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = ["__version__", "client_gen", "client_slo", "client_cmp"]
