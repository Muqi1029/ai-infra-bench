"""Run the DeepSWE coding-agent benchmark through its official Pier runner."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

DEEPSWE_REPOSITORY = "https://github.com/datacurve-ai/deep-swe"
DEFAULT_AGENT = "mini-swe-agent"


def resolve_task_path(dataset_path: str | None) -> Path:
    """Resolve either a DeepSWE repository, tasks directory, or single task."""
    configured_path = dataset_path or os.environ.get("DEEPSWE_TASKS_PATH")
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    else:
        candidates.extend([Path("deep-swe/tasks"), Path("deep-swe")])

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "tasks").is_dir():
            candidate = candidate / "tasks"
        if not candidate.is_dir():
            continue
        if (candidate / "task.toml").is_file() or any(candidate.glob("*/task.toml")):
            return candidate

    requested = configured_path or "./deep-swe/tasks"
    raise ValueError(
        f"DeepSWE task directory not found or invalid: {requested!r}. "
        f"Clone {DEEPSWE_REPOSITORY} and pass its tasks directory with "
        "--dataset-path, or set DEEPSWE_TASKS_PATH."
    )


def find_pier() -> str:
    pier = shutil.which("pier")
    if pier is None:
        raise RuntimeError(
            "DeepSWE requires Pier >= 0.3.0. Install it with "
            "`uv tool install --python 3.12 datacurve-pier`, or use the deepswe "
            "extra when AIB itself runs on Python 3.12."
        )
    return pier


def build_command(runtime_args, task_path: Path, pier: str) -> list[str]:
    command = [pier, "run", "--path", str(task_path)]

    if runtime_args.config:
        command.extend(["--config", runtime_args.config])

    agent = runtime_args.deepswe_agent
    if agent or not runtime_args.config:
        command.extend(["--agent", agent or DEFAULT_AGENT])
    if runtime_args.model:
        command.extend(["--model", runtime_args.model])
    if runtime_args.deepswe_environment:
        command.extend(["--env", runtime_args.deepswe_environment])
    if runtime_args.max_concurrency is not None:
        command.extend(["--n-concurrent", str(runtime_args.max_concurrency)])
    if runtime_args.repeat is not None:
        command.extend(["--n-attempts", str(runtime_args.repeat)])
    if runtime_args.num_questions is not None:
        command.extend(["--n-tasks", str(runtime_args.num_questions)])
        seed = runtime_args.deepswe_sample_seed
        command.extend(["--sample-seed", str(0 if seed is None else seed)])
    elif runtime_args.deepswe_sample_seed is not None:
        command.extend(["--sample-seed", str(runtime_args.deepswe_sample_seed)])
    if runtime_args.deepswe_jobs_dir:
        command.extend(["--jobs-dir", runtime_args.deepswe_jobs_dir])
    if runtime_args.deepswe_env_file:
        command.extend(["--env-file", runtime_args.deepswe_env_file])
    if runtime_args.deepswe_yes:
        command.append("--yes")

    return command


def build_environment(runtime_args, environ: Mapping[str, str]) -> dict[str, str]:
    child_env = dict(environ)

    if runtime_args.api_key != "EMPTY":
        child_env["OPENAI_API_KEY"] = runtime_args.api_key

    if runtime_args.base_url:
        base_url = runtime_args.base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        child_env["OPENAI_BASE_URL"] = base_url
        child_env["OPENAI_API_BASE"] = base_url
        child_env.setdefault("OPENAI_API_KEY", "EMPTY")

    return child_env


class DeepSWERuntime:
    """Translate aib eval-dataset arguments into an official Pier job."""

    def __init__(self, runtime_args):
        self.runtime_args = runtime_args

    def run(self) -> int:
        task_path = resolve_task_path(self.runtime_args.dataset_path)
        command = build_command(self.runtime_args, task_path, find_pier())
        logger.info("Running DeepSWE with Pier: %s", shlex.join(command))

        result = subprocess.run(
            command,
            env=build_environment(self.runtime_args, os.environ),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DeepSWE Pier job failed with exit code {result.returncode}"
            )
        return result.returncode
