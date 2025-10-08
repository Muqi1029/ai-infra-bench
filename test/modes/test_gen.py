import os
import tempfile
import unittest
from glob import glob

from utils import client_cmd_str, input_features, output_metrics, server_cmd_str

from ai_infra_bench.client import client_gen
from ai_infra_bench.sgl import gen_bench
from ai_infra_bench.utils import CSV_NAME, FULL_DATA_JSON_PATH, TABLE_NAME


class TestSGLGen(unittest.TestCase):

    def check_output_content(self, output_dir, expected_files=None):
        expected_files = expected_files or [FULL_DATA_JSON_PATH, TABLE_NAME, CSV_NAME]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(output_dir, f)), f"Missing {f}")

    def run_single_cmd(self, server_cmds, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            gen_bench(
                server_cmds=server_cmds,
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                **kwargs,
            )
            self.check_output_content(output_dir=output_dir)

    def test_basic(self):
        self.run_single_cmd(server_cmd_str, client_cmds=client_cmd_str)

    def test_client_list(self):
        self.run_single_cmd(server_cmd_str, client_cmds=[client_cmd_str] * 3)

    def test_list(self):
        self.run_single_cmd([server_cmd_str], client_cmds=[client_cmd_str] * 3)

    def test_n(self):
        self.run_single_cmd([server_cmd_str], client_cmds=[client_cmd_str] * 3, n=3)

    @unittest.expectedFailure
    def test_length_fail(self):
        self.run_single_cmd([server_cmd_str] * 2, client_cmds=[client_cmd_str] * 3)

    def test_length(self):
        self.run_single_cmd([server_cmd_str] * 2, client_cmds=[[client_cmd_str]] * 2)

    def test_server_labels(self):
        self.run_single_cmd(server_cmd_str, client_cmd_str, server_labels="ServerLabel")
        self.run_single_cmd(
            server_cmd_str, client_cmd_str, server_labels=["ServerLabel"]
        )

    def test_client_labels(self):
        self.run_single_cmd(server_cmd_str, client_cmd_str, client_labels="ClientLabel")
        self.run_single_cmd(
            server_cmd_str, client_cmd_str, client_labels=["ClientLabel"]
        )
        self.run_single_cmd(
            server_cmd_str, [client_cmd_str] * 2, client_labels=["ClientLabel"] * 2
        )
        self.run_single_cmd(
            server_cmd_str, [[client_cmd_str] * 2], client_labels=[["ClientLabel"] * 2]
        )
        self.run_single_cmd(
            server_cmds=[server_cmd_str] * 2,
            client_cmds=[[client_cmd_str] * 2] * 2,
            client_labels=[["ClientLabel"] * 2] * 2,
        )


class TestClientGen(unittest.TestCase):

    def run_single_cmd(self, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
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
                input_features=input_features,
                output_metrics=output_metrics,
            )

    def test_output_file(self):
        with self.assertRaises(AssertionError):
            client_cmd = f"{client_cmd_str} --output-file output.jsonl"
            client_gen(
                client_cmds=client_cmd,
                input_features=input_features,
                output_metrics=output_metrics,
            )

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_plot=True,
            )
            filepaths = glob(f"{output_dir}/*.html")
            self.assertEqual(len(filepaths), 0)

    def test_disable_md_table(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/table.md"))


if __name__ == "__main__":
    unittest.main()
