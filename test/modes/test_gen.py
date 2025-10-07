import os
import tempfile
import unittest
from glob import glob

from utils import client_cmd_str

from ai_infra_bench.client import client_gen
from ai_infra_bench.utils import CSV_NAME, FULL_DATA_JSON_PATH, TABLE_NAME


class TestSGLGen(unittest.TestCase):
    pass


class TestGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_features = [
            "random_input_len",
            "random_output_len",
            "request_rate",
            "max_concurrency",
        ]
        cls.output_metrics = [
            "p99_ttft_ms",
            "p99_tpot_ms",
            "p99_itl_ms",
            "output_throughput",
            "p99_e2e_latency_ms",
        ]

    def run_single_cmd(self, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmds,
                input_features=self.input_features,
                output_metrics=self.output_metrics,
                output_dir=output_dir,
                **kwargs,
            )
            self.check_output_content(output_dir)

    def check_output_content(self, output_dir, expected_files=None):
        expected_files = expected_files or [FULL_DATA_JSON_PATH, TABLE_NAME, CSV_NAME]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(output_dir, f)), f"Missing {f}")

    def test_str(self):
        self.run_single_cmd(client_cmd_str)

    def test_list_str(self):
        self.run_single_cmd([client_cmd_str])

    def test_n(self):
        self.run_single_cmd(client_cmd_str, n=3)

    def test_client_cmds_type(self):
        with self.assertRaises(ValueError):
            client_gen(
                client_cmds=[[client_cmd_str]],
                input_features=self.input_features,
                output_metrics=self.output_metrics,
            )

    def test_output_file(self):
        with self.assertRaises(AssertionError):
            client_cmd = f"{client_cmd_str} --output-file output.jsonl"
            client_gen(
                client_cmds=client_cmd,
                input_features=self.input_features,
                output_metrics=self.output_metrics,
            )

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=self.input_features,
                output_metrics=self.output_metrics,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=self.input_features,
                output_metrics=self.output_metrics,
                output_dir=output_dir,
                disable_plot=True,
            )
            filepaths = glob(f"{output_dir}/*.html")
            self.assertEqual(len(filepaths), 0)

    def test_disable_md_table(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=self.input_features,
                output_metrics=self.output_metrics,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/table.md"))


if __name__ == "__main__":
    unittest.main()
