import logging

from ai_infra_bench.client import client_gen, client_slo
from ai_infra_bench.version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(levelname)s - %(message)s (%(asctime)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = ["__version__", "client_gen", "client_slo"]
