import os
import tempfile
import unittest
from glob import glob
from typing import Dict

from utils import input_features, output_metrics

client_cmd_str = """
python -m sglang.bench_serving \
        --base-url http://localhost:8888
		--backend sglang-oai
        --tokenizer Qwen/Qwen3-0.6B
        --tokenizer Qwen/Qwen3-0.6B
		--dataset-name random
		--random-range-ratio 1
		--random-input-len 1200
		--random-output-len 800
		--num-prompt 40
"""
request_rates = (1, 10)


def check_slo(item: Dict) -> bool:
    return (
        item["p99_ttft_ms"] < 3000
        and item["p99_tpot_ms"] < 100
        and item["p99_itl_ms"] < 100
    )


from ai_infra_bench.client import client_slo
from ai_infra_bench.sgl import slo_bench
from ai_infra_bench.utils import CSV_NAME, FULL_DATA_JSON_PATH, TABLE_NAME


class TestSGLSlo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server_cmd = f"""
        python -m sglang.launch_server
            --model-path Qwen/Qwen3-0.6B
            --port 8888
        """

    def run_single_cmd(
        self,
        server_cmds,
        client_cmds,
        request_rates=request_rates,
        check_slo=check_slo,
        **kwargs,
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            slo_bench(
                server_cmds=server_cmds,
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
                output_dir=output_dir,
                **kwargs,
            )
            self.check_output_content(output_dir)

    def check_output_content(self, output_dir, expected_files=None):
        expected_files = expected_files or [FULL_DATA_JSON_PATH, TABLE_NAME, CSV_NAME]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(output_dir, f)), f"Missing {f}")

    def test_basic(self):
        self.run_single_cmd(self.server_cmd, client_cmds=client_cmd_str)

    def test_client_list(self):
        self.run_single_cmd(self.server_cmd, client_cmds=[client_cmd_str])

    def test_full_list(self):
        self.run_single_cmd([self.server_cmd], client_cmds=[client_cmd_str])

    def test_full_list_length(self):
        self.run_single_cmd([self.server_cmd] * 2, client_cmds=[client_cmd_str] * 2)

    def test_n(self):
        self.run_single_cmd(
            [self.server_cmd] * 2, client_cmds=[client_cmd_str] * 2, n=3
        )

    @unittest.expectedFailure
    def test_length_fail(self):
        self.run_single_cmd([self.server_cmd] * 2, client_cmds=[client_cmd_str] * 3)

    def test_server_labels(self):
        self.run_single_cmd(
            self.server_cmd, client_cmd_str, server_labels="ServerLabel"
        )
        self.run_single_cmd(
            self.server_cmd, client_cmd_str, server_labels=["ServerLabel"]
        )

    def test_client_labels(self):
        self.run_single_cmd(
            server_cmds=self.server_cmd,
            client_cmds=client_cmd_str,
            client_labels="ClientLabel",
        )
        self.run_single_cmd(
            server_cmds=self.server_cmd,
            client_cmds=client_cmd_str,
            client_labels=["ClientLabel"],
        )
        self.run_single_cmd(
            [self.server_cmd] * 2,
            [client_cmd_str] * 2,
            client_labels=["ClientLabel"] * 2,
        )

    def test_server_client_labels(self):
        self.run_single_cmd(
            server_cmds=[self.server_cmd] * 2,
            client_cmds=[client_cmd_str] * 2,
            server_labels="ServerLabel",
            client_labels="ClientLabel",
        )


class TestClientSlo(unittest.TestCase):

    def run_single_cmd(
        self,
        client_cmds,
        request_rates=request_rates,
        check_slo=check_slo,
        **kwargs,
    ):
        with tempfile.TemporaryDirectory() as output_dir:
            client_slo(
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
                output_dir=output_dir,
                **kwargs,
            )
            self.check_output_content(output_dir)

    def check_output_content(self, output_dir, expected_files=None):
        expected_files = expected_files or [FULL_DATA_JSON_PATH, TABLE_NAME, CSV_NAME]
        for f in expected_files:
            self.assertTrue(os.path.exists(os.path.join(output_dir, f)), f"Missing {f}")

    def test_str(self):
        self.run_single_cmd(
            client_cmds=client_cmd_str,
        )

    def test_list_str(self):
        self.run_single_cmd([client_cmd_str])

    def test_list_str_complex(self):
        self.run_single_cmd(
            [client_cmd_str] * 3, request_rates=(1, 2), client_labels=["label"] * 3
        )
        self.run_single_cmd(
            [client_cmd_str] * 3, request_rates=(1, 2), client_labels="label"
        )
        self.run_single_cmd(
            [client_cmd_str] * 3, request_rates=[(1, 2)] * 3, client_labels="label"
        )
        self.run_single_cmd(
            [client_cmd_str] * 3,
            request_rates=[(1, 2)] * 3,
            client_labels="label",
            check_slo=[check_slo] * 3,
        )

    @unittest.expectedFailure
    def test_list_str_complex_fail_check_slo(self):
        self.run_single_cmd(
            client_cmds=[client_cmd_str] * 3,
            request_rates=[(1, 2)] * 3,
            client_labels="label",
            check_slo=[check_slo] * 2,
        )

    @unittest.expectedFailure
    def test_list_str_complex_fail_request_rates(self):
        self.run_single_cmd(
            client_cmds=[client_cmd_str] * 3,
            request_rates=[(1, 2)] * 2,
            client_labels="label",
            check_slo=[check_slo] * 3,
        )

    @unittest.expectedFailure
    def test_list_str_complex_fail_client_labels(self):
        self.run_single_cmd(
            client_cmds=[client_cmd_str] * 3,
            request_rates=[(1, 2)] * 3,
            client_labels=["label"] * 2,
            check_slo=[check_slo] * 3,
        )

    def test_n(self):
        self.run_single_cmd(client_cmd_str, n=3)

    def test_client_cmds_type(self):
        with self.assertRaises(ValueError):
            client_slo(
                client_cmds=[[client_cmd_str]],
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
            )

    def test_output_file(self):
        with self.assertRaises(AssertionError):
            client_cmd = f"{client_cmd_str} --output-file output.jsonl"
            client_slo(
                client_cmds=client_cmd,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
            )

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_slo(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_slo(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
                output_dir=output_dir,
                disable_plot=True,
            )
            filepaths = glob(f"{output_dir}/*.html")
            self.assertEqual(len(filepaths), 0)

    def test_disable_md_table(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_slo(
                client_cmds=client_cmd_str,
                input_features=input_features,
                output_metrics=output_metrics,
                request_rates=request_rates,
                check_slo=check_slo,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/table.md"))


if __name__ == "__main__":
    unittest.main()
