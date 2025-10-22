import os
import tempfile
import unittest
from glob import glob

from utils import check_output_content, input_features, output_metrics

from ai_infra_bench.client import client_gen
from ai_infra_bench.sgl import gen_bench
from ai_infra_bench.utils import CSV_NAME, TABLE_NAME, WARMUP_FILE


class TestClientGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cmd = """
        python -m sglang.bench_serving \
                --base-url http://localhost:8888
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

    #################### Basic Usage Cases ##################
    def run_single_cmd(self, client_cmds, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=client_cmds,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                **kwargs,
            )
            check_output_content(output_dir)

    def test_single_run(self):
        self.run_single_cmd(self.client_cmd)

    def test_multiple_run(self):
        self.run_single_cmd([self.client_cmd])
        self.run_single_cmd([self.client_cmd] * 3)

    def test_n(self):
        self.run_single_cmd(self.client_cmd, n=3)
        self.run_single_cmd([self.client_cmd], n=3)
        self.run_single_cmd([self.client_cmd] * 3, n=3)
        self.run_single_cmd([self.client_cmd] * 3, n=3, only_last=True)

    ####################### Labels ###################
    def test_server_label(self):
        self.run_single_cmd(client_cmds=self.client_cmd, server_label="server_label")
        self.run_single_cmd(client_cmds=self.client_cmd, server_label=["server_label"])
        self.run_single_cmd(
            client_cmds=[self.client_cmd], server_label=["server_label"]
        )
        self.run_single_cmd(
            client_cmds=[self.client_cmd] * 3, server_label=["server_label"]
        )

    def test_client_labels(self):
        self.run_single_cmd(client_cmds=self.client_cmd, client_labels="client_label")
        self.run_single_cmd(
            client_cmds=[self.client_cmd], client_labels=["client_label"]
        )
        self.run_single_cmd(
            client_cmds=[self.client_cmd] * 3, client_labels=["client_label"] * 3
        )

    #################### Expected Failed ###############
    @unittest.expectedFailure
    def test_server_label(self):
        self.run_single_cmd(
            client_cmds=self.client_cmd, server_label=["server_label"] * 2
        )

    @unittest.expectedFailure
    def test_client_labels(self):
        self.run_single_cmd(
            client_cmds=[self.client_cmd], client_labels=["client_label"] * 3
        )

    @unittest.expectedFailure
    def test_client_cmds_type(self):
        client_gen(
            client_cmds=[[self.client_cmd]],
            input_features=input_features,
            output_metrics=output_metrics,
        )

    @unittest.expectedFailure
    def test_output_file(self):
        client_cmd = f"{self.client_cmd} --output-file output.jsonl"
        client_gen(
            client_cmds=client_cmd,
            input_features=input_features,
            output_metrics=output_metrics,
        )

    ########### Test Disable ###################
    def test_disable_warmup(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=self.client_cmd,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_warmup=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{WARMUP_FILE}"))

    def test_disable_csv(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=self.client_cmd,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_csv=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{CSV_NAME}"))

    def test_disable_plot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            client_gen(
                client_cmds=self.client_cmd,
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
                client_cmds=self.client_cmd,
                input_features=input_features,
                output_metrics=output_metrics,
                output_dir=output_dir,
                disable_table=True,
            )
            self.assertTrue(not os.path.exists(f"{output_dir}/{TABLE_NAME}"))


class TestSGLGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_cmd = f"""
        python -m sglang.launch_server
            --model-path Qwen/Qwen3-0.6B
            --port 8888
        """
        cls.client_cmd = """
        python -m sglang.bench_serving \
                --base-url http://localhost:8888
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

    ########### basic use cases #########
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
            check_output_content(output_dir=output_dir)

    def test_single_run(self):
        self.run_single_cmd(self.server_cmd, client_cmds=self.client_cmd)

    def test_multiple_run(self):
        self.run_single_cmd(self.server_cmd, client_cmds=[self.client_cmd] * 3)
        self.run_single_cmd([self.server_cmd], client_cmds=[self.client_cmd] * 3)
        self.run_single_cmd([self.server_cmd], client_cmds=[[self.client_cmd] * 3])
        self.run_single_cmd(
            [self.server_cmd] * 2, client_cmds=[[self.client_cmd] * 3 for _ in range(2)]
        )

    def test_n(self):
        self.run_single_cmd(self.server_cmd, client_cmds=[self.client_cmd] * 3, n=3)
        self.run_single_cmd(
            server_cmds=[self.server_cmd], client_cmds=[self.client_cmd] * 3, n=3
        )
        self.run_single_cmd(
            server_cmds=[self.server_cmd] * 2,
            client_cmds=[[self.client_cmd] * 3 for _ in range(2)],
            n=3,
        )
        self.run_single_cmd(
            server_cmds=[self.server_cmd] * 2,
            client_cmds=[[self.client_cmd] * 3 for _ in range(2)],
            n=3,
            only_last=True,
        )

    ################# expected failures ################
    @unittest.expectedFailure
    def test_length_fail(self):
        self.run_single_cmd([self.server_cmd] * 2, client_cmds=[self.client_cmd] * 3)

    ################# server & client labels ###############
    def test_server_labels(self):
        self.run_single_cmd(
            self.server_cmd, self.client_cmd, server_labels="ServerLabel"
        )
        self.run_single_cmd(
            self.server_cmd, self.client_cmd, server_labels=["ServerLabel"]
        )

    def test_client_labels(self):
        self.run_single_cmd(
            self.server_cmd, self.client_cmd, client_labels="ClientLabel"
        )
        self.run_single_cmd(
            self.server_cmd, self.client_cmd, client_labels=["ClientLabel"]
        )
        self.run_single_cmd(
            self.server_cmd, [self.client_cmd] * 2, client_labels=["ClientLabel"] * 2
        )
        self.run_single_cmd(
            self.server_cmd,
            [[self.client_cmd] * 2],
            client_labels=[["ClientLabel"] * 2],
        )
        self.run_single_cmd(
            server_cmds=[self.server_cmd] * 2,
            client_cmds=[[self.client_cmd] * 2] * 2,
            client_labels=[["ClientLabel"] * 2] * 2,
        )


if __name__ == "__main__":
    unittest.main()
