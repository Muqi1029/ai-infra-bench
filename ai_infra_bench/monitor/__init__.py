from typing import Any


def auto_register(*args: Any, **kwargs: Any):
    from ai_infra_bench.monitor.launcher import auto_register as _auto_register

    return _auto_register(*args, **kwargs)


__all__ = ["auto_register"]
