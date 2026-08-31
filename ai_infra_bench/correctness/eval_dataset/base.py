import asyncio
import json
import logging
from asyncio import Semaphore
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple, TypedDict

from aiohttp import ClientSession
from tqdm import tqdm

from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.draw import format_mean, format_percentile, print_table
from ai_infra_bench.utils.req import extract_response_metrics

logger = logging.getLogger(__name__)

DATASET_CHOICES = ["gsm8k", "aime25", "gpqa", "hle", "constrained_decoding"]
DEEPSWE_EVAL = "deepswe"
AGENT_DATASET_CHOICES = [DEEPSWE_EVAL]
TOKEN_METRICS = (
    ("reasoning_tokens", "Reasoning tokens"),
    ("prompt_tokens", "Prompt tokens"),
    ("completion_tokens", "Completion tokens"),
)


class TokenUsageStats(TypedDict):
    total: int
    mean: str
    p50: str
    p90: str
    p99: str


class EvalRuntime:
    def __init__(self, runtime_args):
        self.endpoint_url = runtime_args.base_url.rstrip("/") + "/v1/chat/completions"
        self.max_concurrency = runtime_args.max_concurrency
        self.api_key = runtime_args.api_key
        self.repeat = runtime_args.repeat
        self.num_questions = runtime_args.num_questions

        self.sem: Semaphore | None = None
        self.session: ClientSession | None = None
        self.evals: List[Eval] = [
            Eval.create_from_name(
                eval_name,
                config_path=runtime_args.config,
                dataset_path=runtime_args.dataset_path,
                num_shots=runtime_args.num_shots,
            )
            for eval_name in runtime_args.evals
        ]
        self.override_payload = {}
        if model := getattr(runtime_args, "model", None):
            self.override_payload["model"] = model
        if runtime_args.override_payload:
            self.override_payload.update(json.loads(runtime_args.override_payload))
            logging.info(f"[All] Override Payload: {self.override_payload}")
        self.maybe_truncate_eval()

    def maybe_truncate_eval(self):
        for eval in self.evals:
            eval.maybe_truncate(self.num_questions)

    async def _run_one(self, eval: "Eval", payload: Dict, answer: Any, pbar: tqdm):
        assert self.session is not None and self.sem is not None
        async with self.sem:
            try:
                async with self.session.post(
                    self.endpoint_url, json=payload
                ) as response:
                    body = await response.json(content_type=None)
                    if response.status != 200:
                        logger.error(
                            "HTTP %s from %s: %s",
                            response.status,
                            self.endpoint_url,
                            body,
                        )
                        eval.add_failed_result(body, payload)
                        return

                    eval.eval(body, answer, payload)
            except Exception:
                logger.exception("Request failed for eval=%s", eval.name)
                eval.add_failed_result(None, payload)
            finally:
                pbar.update(1)

    async def run(self):
        # aiohttp connector / semaphore must be created inside a running loop.
        self.sem = Semaphore(self.max_concurrency)
        self.session = _create_bench_client_session(self.max_concurrency, self.api_key)

        total = self.repeat * sum(e.get_length() for e in self.evals)
        pbar = tqdm(total=total)
        try:
            async with self.session:
                for eval in self.evals:
                    for i in range(self.repeat):
                        if self.repeat == 1:
                            pbar.set_description(f"Eval {eval.name}")
                        else:
                            pbar.set_description(
                                f"Eval {eval.name} [{i + 1}/{self.repeat}]"
                            )

                        tasks = [
                            asyncio.create_task(
                                self._run_one(eval, payload, answer, pbar)
                            )
                            for payload, answer in eval.get_payload_and_answer(
                                self.override_payload
                            )
                        ]
                        round_start_time = perf_counter()
                        await asyncio.gather(*tasks)
                        round_duration_s = max(perf_counter() - round_start_time, 1e-9)

                        for title, rows in eval.build_summary_tables(
                            round_number=i + 1,
                            duration_s=round_duration_s,
                            max_concurrency=self.max_concurrency,
                        ):
                            print_table(title, rows)
        finally:
            pbar.close()
            self.session = None
            self.sem = None


@dataclass
class EvalResult:
    response: Any
    payload: Dict = field(default_factory=dict)
    is_right: bool = False
    is_failed: bool = False
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Eval:
    name: str = ""
    results: List[EvalResult] = field(default_factory=list)

    def get_length(self) -> int:
        raise NotImplementedError()

    def get_payload_and_answer(self) -> Iterable[Tuple[Dict, Any]]:
        raise NotImplementedError()

    def _eval(self, response_content, answer, payload=None):
        raise NotImplementedError()

    @staticmethod
    def _token_usage(response_content: Any) -> Dict[str, int]:
        metrics = extract_response_metrics(response_content)
        return {
            key: int(metrics.get(key) or 0)
            for key in ("reasoning_tokens", "prompt_tokens", "completion_tokens")
        }

    def eval(self, response_content, answer, payload):
        token_usage = self._token_usage(response_content)
        self.results.append(
            EvalResult(
                response=response_content,
                payload=payload,
                is_right=self._eval(response_content, answer, payload),
                **token_usage,
            )
        )

    def add_failed_result(self, response_content, payload):
        token_usage = self._token_usage(response_content)
        self.results.append(
            EvalResult(
                response=response_content,
                payload=payload,
                is_right=False,
                is_failed=True,
                **token_usage,
            )
        )

    def token_usage(self) -> Dict[str, TokenUsageStats]:
        successful_results = [result for result in self.results if not result.is_failed]
        token_usage = {}
        for key, _ in TOKEN_METRICS:
            values = [getattr(result, key) for result in successful_results]
            token_usage[key] = {
                "total": sum(values),
                "mean": format_mean(values),
                "p50": format_percentile(values, 50),
                "p90": format_percentile(values, 90),
                "p99": format_percentile(values, 99),
            }
        return token_usage

    def build_summary_tables(
        self, round_number: int, duration_s: float, max_concurrency: int
    ) -> List[Tuple[str, List[List[Any]]]]:
        total_requests = len(self.results)
        successful_requests = sum(not result.is_failed for result in self.results)
        token_usage = self.token_usage()
        tps = token_usage["completion_tokens"]["total"] / max(duration_s, 1e-9)
        correct_rate, wrong_rate, failed_rate = self.summary()

        summary_rows = [
            ["Metric", "Value"],
            ["Evaluation", self.name],
            ["Round", str(round_number)],
            ["Total requests", str(total_requests)],
            ["Successful requests", str(successful_requests)],
            ["Failed requests", str(total_requests - successful_requests)],
            ["Correct rate", f"{correct_rate:.2%}"],
            ["Wrong rate", f"{wrong_rate:.2%}"],
            ["Failed rate", f"{failed_rate:.2%}"],
            ["Max concurrency", str(max_concurrency)],
            ["Duration", f"{duration_s:.2f} s"],
            ["TPS", f"{tps:.2f} tokens/s"],
            *[
                [
                    f"Total {label.lower()}",
                    f"{token_usage[key]['total']} tokens",
                ]
                for key, label in TOKEN_METRICS
            ],
        ]
        token_rows = [
            ["Metric", "Mean", "P50", "P90", "P99", "Unit"],
            *[
                [
                    label,
                    token_usage[key]["mean"],
                    token_usage[key]["p50"],
                    token_usage[key]["p90"],
                    token_usage[key]["p99"],
                    "tokens",
                ]
                for key, label in TOKEN_METRICS
            ],
        ]
        return [
            ("Evaluation Summary", summary_rows),
            ("Token Metrics", token_rows),
        ]

    def summary(self):
        if not self.results:
            return 0.0, 0.0, 0.0
        total = len(self.results)
        correct_count = sum(
            1
            for eval_result in self.results
            if eval_result.is_right and not eval_result.is_failed
        )
        failed_count = sum(1 for eval_result in self.results if eval_result.is_failed)
        wrong_count = total - correct_count - failed_count
        correct_rate = correct_count / total
        wrong_rate = wrong_count / total
        failed_rate = failed_count / total
        self.results = []
        return correct_rate, wrong_rate, failed_rate

    @classmethod
    def create_from_name(
        cls,
        name: str,
        config_path: str | None = None,
        dataset_path: str | None = None,
        num_shots: int | None = None,
    ):
        # Convention: name "gsm8k" -> eval_dataset/gsm8k.py -> *Eval subclass
        import importlib

        module_name = name.lower().replace("-", "_")
        try:
            module_path = f"ai_infra_bench.correctness.eval_dataset.{module_name}"
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Failed to import {module_path}. This is mostly due to this eval is not supported yet"
            ) from e

        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, cls) and obj is not cls:
                kwargs = {}
                if config_path is not None:
                    kwargs["config_path"] = config_path
                if dataset_path is not None:
                    kwargs["dataset_path"] = dataset_path
                if num_shots is not None:
                    kwargs["num_shots"] = num_shots
                return obj(name, **kwargs)

        raise ValueError(f"No Eval subclass found in module for name={name!r}")
