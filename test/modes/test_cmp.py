import os
import tempfile
import unittest
from glob import glob

from utils import check_output_content, input_features, output_metrics

from ai_infra_bench.client import client_cmp
from ai_infra_bench.sgl import cmp_bench
from ai_infra_bench.utils import CSV_NAME, TABLE_NAME, WARMUP_FILE, ServerAccessInfo


class TestClientCmp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_access_info = ServerAccessInfo(base_url="http://127.0.0.1:8000")
        cls.client_cmds = """
python -m sglang.bench_serving \
		--backend sglang-oai
        --tokenizer Qwen/Qwen3-0.6B
        --tokenizer Qwen/Qwen3-0.6B
		--dataset-name random
		--random-range-ratio 1
		--random-input-len 1200
		--random-output-len 800
		--request-rate 10
		--max-concurrency 10
		--num-prompt 40
            """

    def run_single_cmd(self, server_access_info, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            client_cmp(
                server_access_info=server_access_info,
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                **kwargs,
            )
            check_output_content(
                output_dir, expected_files=[CSV_NAME, TABLE_NAME, WARMUP_FILE]
            )

    ################## BASIC RUN ##########################
    def test_single_run(self):
        self.run_single_cmd(self.server_access_info, self.client_cmds)

    def test_multiple_run(self):
        self.run_single_cmd([self.server_access_info] * 2, [self.client_cmds] * 3)

        self.run_single_cmd([self.server_access_info] * 2, self.client_cmds)

        self.run_single_cmd(self.server_access_info, [self.client_cmds] * 2)

    def test_n_run(self):
        self.run_single_cmd(self.server_access_info, self.client_cmds, n=3)

        self.run_single_cmd([self.server_access_info] * 2, [self.client_cmds] * 3, n=3)

        self.run_single_cmd([self.server_access_info] * 2, self.client_cmds, n=3)

        self.run_single_cmd(self.server_access_info, [self.client_cmds] * 2, n=3)

    ################## LABEL SETTING ##########################
    def test_client_labels(self):
        self.run_single_cmd(
            self.server_access_info, self.client_cmds, client_labels=["client1"]
        )

        self.run_single_cmd(
            self.server_access_info, self.client_cmds, client_labels="client1"
        )

        self.run_single_cmd(
            self.server_access_info, [self.client_cmds] * 2, client_labels="client1"
        )

        self.run_single_cmd(
            self.server_access_info,
            [self.client_cmds] * 2,
            client_labels=["client1"] * 2,
        )

    @unittest.expectedFailure
    def test_failed_client_labels(self):
        self.run_single_cmd(
            self.server_access_info,
            [self.client_cmds] * 2,
            client_labels=["client1"] * 3,
        )

    def test_server_labels(self):
        self.run_single_cmd(
            self.server_access_info, self.client_cmds, server_labels="server_label"
        )
        self.run_single_cmd(
            self.server_access_info, self.client_cmds, server_labels=["server_label"]
        )
        self.run_single_cmd(
            [self.server_access_info] * 2,
            self.client_cmds,
            server_labels=["server_label"] * 2,
        )

    @unittest.expectedFailure
    def test_failed_server_labels(self):
        self.run_single_cmd(
            [self.server_access_info] * 2,
            self.client_cmds,
            server_labels=["server_label"] * 3,
        )

    ################## EXPECTED FAIL ###########################
    @unittest.expectedFailure
    def test_failed_host(self):
        self.run_single_cmd(
            self.server_access_info, self.client_cmds + " --host 127.0.0.1"
        )

    @unittest.expectedFailure
    def test_failed_port(self):
        self.run_single_cmd(self.server_access_info, self.client_cmds + " --port 8888")

    @unittest.expectedFailure
    def test_failed_base_url(self):
        self.run_single_cmd(
            self.server_access_info,
            self.client_cmds + " --base-url http://127.0.0.1:8888",
        )

    @unittest.expectedFailure
    def test_failed_output_file(self):
        self.run_single_cmd(
            self.server_access_info, self.client_cmds + " --output-file output.jsonl"
        )

    ################## DISABLE FEATURE ##########################
    def test_disable_warmup(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_cmp(
                server_access_info=self.server_access_info,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_warmup=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{WARMUP_FILE}"))

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_cmp(
                server_access_info=self.server_access_info,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_md_table(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_cmp(
                server_access_info=self.server_access_info,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{TABLE_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_cmp(
                server_access_info=self.server_access_info,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_plot=True,
            )
            filepaths = glob(f"{output_dir}/*.html")
            self.assertEqual(len(filepaths), 0)


class TestSGLCmp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cmds = """
        python -m sglang.bench_serving
                --backend sglang-oai
                --tokenizer Qwen/Qwen3-0.6B
                --tokenizer Qwen/Qwen3-0.6B
                --dataset-name random
                --random-range-ratio 1
                --random-input-len 1200
                --random-output-len 800
                --request-rate 10
                --max-concurrency 10
                --num-prompt 40
            """
        cls.server_cmds = f"""
        python -m sglang.launch_server
            --model-path Qwen/Qwen3-0.6B
            --port 8888
        """

    def run_single_cmd(self, server_cmds, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            cmp_bench(
                server_cmds=server_cmds,
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                port=8888,
                **kwargs,
            )
            check_output_content(
                output_dir, expected_files=[CSV_NAME, TABLE_NAME, WARMUP_FILE]
            )

    ################## BASIC RUN ##########################
    def test_single_run(self):
        self.run_single_cmd(self.server_cmds, self.client_cmds)

    def test_multiple_run(self):
        self.run_single_cmd([self.server_cmds] * 2, [self.client_cmds] * 3)

        self.run_single_cmd([self.server_cmds] * 2, self.client_cmds)

        self.run_single_cmd(self.server_cmds, [self.client_cmds] * 2)

    def test_n_run(self):
        self.run_single_cmd(self.server_cmds, self.client_cmds, n=3)

        self.run_single_cmd([self.server_cmds] * 2, [self.client_cmds] * 3, n=3)

        self.run_single_cmd([self.server_cmds] * 2, self.client_cmds, n=3)

        self.run_single_cmd(self.server_cmds, [self.client_cmds] * 2, n=3)

    ################## LABEL SETTING ##########################
    def test_client_labels(self):
        self.run_single_cmd(
            self.server_cmds, self.client_cmds, client_labels=["client1"]
        )

        self.run_single_cmd(self.server_cmds, self.client_cmds, client_labels="client1")

        self.run_single_cmd(
            self.server_cmds, [self.client_cmds] * 2, client_labels="client1"
        )

        self.run_single_cmd(
            self.server_cmds,
            [self.client_cmds] * 2,
            client_labels=["client1"] * 2,
        )

    @unittest.expectedFailure
    def test_failed_client_labels(self):
        self.run_single_cmd(
            self.server_cmds,
            [self.client_cmds] * 2,
            client_labels=["client1"] * 3,
        )

    def test_server_labels(self):
        self.run_single_cmd(
            self.server_cmds, self.client_cmds, server_labels="server_label"
        )
        self.run_single_cmd(
            self.server_cmds, self.client_cmds, server_labels=["server_label"]
        )
        self.run_single_cmd(
            [self.server_cmds] * 2,
            self.client_cmds,
            server_labels=["server_label"] * 2,
        )

    @unittest.expectedFailure
    def test_failed_server_labels(self):
        self.run_single_cmd(
            [self.server_cmds] * 2,
            self.client_cmds,
            server_labels=["server_label"] * 3,
        )

    ################## EXPECTED FAIL ###########################
    @unittest.expectedFailure
    def test_failed_output_file(self):
        self.run_single_cmd(
            self.server_cmds, self.client_cmds + " --output-file output.jsonl"
        )

    ################## DISABLE FEATURE ##########################
    def test_disable_warmup(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cmp_bench(
                server_cmds=self.server_cmds,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_warmup=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{WARMUP_FILE}"))

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cmp_bench(
                server_cmds=self.server_cmds,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_md_table(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cmp_bench(
                server_cmds=self.server_cmds,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{TABLE_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cmp_bench(
                server_cmds=self.server_cmds,
                client_cmds=self.client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_plot=True,
            )
            filepaths = glob(f"{output_dir}/*.html")
            self.assertEqual(len(filepaths), 0)


if __name__ == "__main__":
    unittest.main()
