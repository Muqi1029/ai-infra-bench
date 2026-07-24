import asyncio
import json
import logging
from asyncio import Semaphore
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from aiohttp import ClientSession
from tqdm import tqdm

from ai_infra_bench.check.common import _create_bench_client_session

logger = logging.getLogger(__name__)

DATASET_CHOICES = ["gsm8k", "aime25", "constrained_decoding"]


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
        if runtime_args.override_payload:
            self.override_payload = json.loads(runtime_args.override_payload)
            logging.info(f"Override Payload: {self.override_payload}")
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
                        await asyncio.gather(*tasks)

                        correct_rate, wrong_rate, failed_rate = eval.summary()
                        logger.info(
                            "%s round %s: correct_rate=%.4f wrong_rate=%.4f failed_rate=%.4f",
                            eval.name,
                            i + 1,
                            correct_rate,
                            wrong_rate,
                            failed_rate,
                        )
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


class Eval:
    name: str = ""
    results: List[EvalResult] = field(default_factory=list)

    def get_length(self) -> int:
        raise NotImplementedError()

    def get_payload_and_answer(self) -> Iterable[Tuple[Dict, Any]]:
        raise NotImplementedError()

    def _eval(self, response_content, answer, payload=None):
        raise NotImplementedError()

    def eval(self, response_content, answer, payload):
        self.results.append(
            EvalResult(
                response=response_content,
                payload=payload,
                is_right=self._eval(response_content, answer, payload),
            )
        )

    def add_failed_result(self, response_content, payload):
        self.results.append(
            EvalResult(
                response=response_content,
                payload=payload,
                is_right=False,
                is_failed=True,
            )
        )

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
            module = importlib.import_module(
                f"ai_infra_bench.check.correctness.eval_dataset.{module_name}"
            )
        except ModuleNotFoundError as e:
            raise ValueError(f"Unknown eval name: {name!r}") from e

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
