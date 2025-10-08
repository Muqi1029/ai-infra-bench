import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ai_infra_bench.utils import (
    FULL_DATA_JSON_PATH,
    add_request_rate,
    enter_decorate,
    read_jsonl,
    run_cmd,
    sort_data_by_key,
)

logger = logging.getLogger(__name__)


@enter_decorate(title="SLO RUN")
def slo_run(
    client_cmd: str,
    request_rate: Tuple[int, int],
    check_slo: Callable,
    n: int,
    output_dir: str,
    label: None | str,
):
    left, right = request_rate

    inner_client_data: List[List[Dict[str, Any]]] = []
    while left <= right:
        mid = (left + right) // 2
        cmd = add_request_rate(client_cmd, mid)

        inner_data: List[Dict] = []
        for run_id in range(n):
            output_file = (
                f"{label}_run{run_id:02d}.jsonl" if n != 1 else f"{label}.jsonl"
            )
            output_file = os.path.join(output_dir, FULL_DATA_JSON_PATH, output_file)
            cmd += f" --output-file {output_file}"
            run_cmd(cmd, is_block=True)
            inner_data.append(read_jsonl(output_file)[-1])

        union_avg_item = {}
        for key in inner_data[0].keys():
            if not inner_data[0][key] or isinstance(inner_data[0][key], str):
                union_avg_item[key] = inner_data[0][key]
            else:
                union_avg_item[key] = np.median([item[key] for item in inner_data])
        if check_slo(union_avg_item):
            left = mid + 1
        else:
            right = mid - 1

        inner_client_data.append(inner_data)

    logger.info(f"\033[92m The maximum concurrency satisfying SLO is {right} \033[0m")

    sorted_inner_client_data = sort_data_by_key(
        key="max_concurrency", data=inner_client_data
    )
    return sorted_inner_client_data, right
