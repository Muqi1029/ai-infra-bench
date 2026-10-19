"""Tests for the DeepSWE Pier integration."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_infra_bench.correctness.eval_dataset.base import (
    AGENT_DATASET_CHOICES,
    DATASET_CHOICES,
    EvalRuntime,
)
from ai_infra_bench.correctness.eval_dataset.deepswe import (
    DeepSWERuntime,
    build_command,
    build_environment,
    resolve_task_path,
)
from ai_infra_bench.correctness.eval_dataset.main import main, parse_args


def runtime_args(**overrides):
    values = {
        "api_key": "EMPTY",
        "base_url": None,
        "config": None,
        "dataset_path": None,
        "deepswe_agent": None,
        "deepswe_environment": None,
        "deepswe_env_file": None,
        "deepswe_jobs_dir": None,
        "deepswe_sample_seed": None,
        "deepswe_yes": False,
        "max_concurrency": 1,
        "model": "openai/test-model",
        "num_questions": None,
        "repeat": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDeepSWE(unittest.TestCase):
    def test_legacy_dataset_defaults_are_preserved(self):
        args = parse_args(["--evals", "gsm8k"])
        self.assertEqual(args.base_url, "http://127.0.0.1:30000")
        self.assertEqual(args.api_key, "JustKeepMe")
        self.assertEqual(args.max_concurrency, 32)
        self.assertEqual(args.repeat, 1)

    def test_dataset_registry_keeps_legacy_choices(self):
        self.assertEqual(DATASET_CHOICES, ["gsm8k", "aime25", "constrained_decoding"])
        self.assertEqual(AGENT_DATASET_CHOICES, ["deepswe"])

    @patch("ai_infra_bench.correctness.eval_dataset.base.Eval.create_from_name")
    def test_model_is_added_to_http_payload_overrides(self, create_eval):
        args = parse_args(
            [
                "--evals",
                "gsm8k",
                "--model",
                "test-model",
                "--override-payload",
                '{"temperature": 0.2}',
            ]
        )
        runtime = EvalRuntime(args)
        self.assertEqual(
            runtime.override_payload,
            {"model": "test-model", "temperature": 0.2},
        )
        create_eval.return_value.maybe_truncate.assert_called_once_with(None)

    def test_parse_args_uses_safe_runner_defaults(self):
        args = parse_args(
            [
                "--evals",
                "deepswe",
                "--dataset-path",
                "/tmp/deep-swe/tasks",
                "--model",
                "openai/test-model",
            ]
        )
        self.assertEqual(args.max_concurrency, 1)
        self.assertEqual(args.repeat, 1)

    def test_parse_args_requires_model_or_config(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--evals", "deepswe", "--dataset-path", "/tmp/tasks"])

    def test_all_keeps_agent_benchmark_explicit(self):
        args = parse_args(["--evals", "all", "gsm8k"])
        self.assertNotIn("deepswe", args.evals)
        self.assertEqual(args.evals, DATASET_CHOICES)
        self.assertEqual(args.max_concurrency, 32)

    def test_resolve_repository_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "tasks" / "task-one"
            task_dir.mkdir(parents=True)
            (task_dir / "task.toml").touch()
            self.assertEqual(
                resolve_task_path(tmpdir), (Path(tmpdir) / "tasks").resolve()
            )

    def test_build_command_maps_common_arguments(self):
        args = runtime_args(
            base_url="http://localhost:9090",
            deepswe_environment="docker",
            deepswe_jobs_dir="results",
            deepswe_sample_seed=7,
            deepswe_yes=True,
            max_concurrency=2,
            num_questions=10,
            repeat=3,
        )
        command = build_command(args, Path("/tasks"), "/usr/bin/pier")
        self.assertEqual(
            command,
            [
                "/usr/bin/pier",
                "run",
                "--path",
                "/tasks",
                "--agent",
                "mini-swe-agent",
                "--model",
                "openai/test-model",
                "--env",
                "docker",
                "--n-concurrent",
                "2",
                "--n-attempts",
                "3",
                "--n-tasks",
                "10",
                "--sample-seed",
                "7",
                "--jobs-dir",
                "results",
                "--yes",
            ],
        )

    def test_build_environment_preserves_secrets_outside_command(self):
        args = runtime_args(
            api_key="secret", base_url="localhost:9090", dataset_path="/tasks"
        )
        env = build_environment(args, {"PATH": "/bin"})
        self.assertEqual(env["OPENAI_API_KEY"], "secret")
        self.assertEqual(env["OPENAI_BASE_URL"], "http://localhost:9090/v1")
        self.assertEqual(env["OPENAI_API_BASE"], "http://localhost:9090/v1")

    def test_main_returns_clean_error_for_invalid_tasks(self):
        with self.assertLogs(level="ERROR") as logs:
            exit_code = main(
                [
                    "--evals",
                    "deepswe",
                    "--dataset-path",
                    "/definitely/missing/deep-swe/tasks",
                    "--model",
                    "openai/test-model",
                ]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("DeepSWE task directory not found", "\n".join(logs.output))

    @patch("ai_infra_bench.correctness.eval_dataset.deepswe.subprocess.run")
    @patch(
        "ai_infra_bench.correctness.eval_dataset.deepswe.find_pier",
        return_value="/usr/bin/pier",
    )
    def test_runtime_runs_pier(self, _find_pier, run):
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task-one"
            task_dir.mkdir()
            (task_dir / "task.toml").touch()
            args = runtime_args(dataset_path=tmpdir)
            self.assertEqual(DeepSWERuntime(args).run(), 0)

        command = run.call_args.args[0]
        self.assertEqual(
            command[:4], ["/usr/bin/pier", "run", "--path", str(Path(tmpdir).resolve())]
        )
        self.assertNotIn("OPENAI_API_KEY=", " ".join(command))
        self.assertEqual(run.call_args.kwargs["check"], False)


if __name__ == "__main__":
    unittest.main()
