import asyncio
import json
import logging
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
from typing import Optional, Sequence

from tqdm import tqdm

from ai_infra_bench.performance.bench import load_requests
from ai_infra_bench.utils.client import _create_bench_client_session
from ai_infra_bench.utils.req import api_url, normalize_payload, sanitize_url

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = ArgumentParser()
    parser.add_argument("--base-url", type=sanitize_url, required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument(
        "--resume-from",
        default=0,
        type=int,
        help="Skip requests before this original input index",
    )
    parser.add_argument("--payload-regex-path", required=True)
    parser.add_argument("--output-file", default="reply_results.jsonl")
    parser.add_argument("--max-concurrency", default=32, type=int)
    parser.add_argument(
        "--filter-constrained-grammar-requests",
        action="store_true",
        help="Filter constrained grammar requests",
    )
    parser.add_argument("--with-ts", action="store_true")
    return parser.parse_args(argv)


def prepare_payload(record):
    raw_payload = record.get("payload", record)
    if not isinstance(raw_payload, dict):
        raise ValueError("payload must be a JSON object")

    payload = deepcopy(raw_payload)
    payload = normalize_payload(payload)
    payload["stream"] = False
    payload["return_token_ids"] = True
    payload.pop("stream_options", None)
    return payload


def extract_token_ids(response_data):
    choice = (response_data.get("choices") or [{}])[0]
    input_ids = choice.get("prompt_token_ids")
    output_ids = choice.get("token_ids")
    if input_ids is None or output_ids is None:
        raise RuntimeError("input_ids & output_ids should not be none")
    return input_ids, output_ids


def select_payloads(payloads, resume_from, num_requests=None):
    if num_requests is not None:
        payloads = payloads[:num_requests]
    if resume_from > len(payloads):
        raise ValueError(
            f"--resume-from={resume_from} exceeds "
            f"the selected request count {len(payloads)}"
        )
    return list(enumerate(payloads[resume_from:], start=resume_from))


async def write_result(file_obj, lock: asyncio.Lock, record):
    async with lock:
        if (
            record["status"] == "success"
            and record["input_ids"] is not None
            and record["output_ids"] is not None
        ):
            input_ids = record["input_ids"]
            output_ids = record["output_ids"]
            output = {
                "input_ids": input_ids + output_ids,
                "loss_mask": [0] * len(input_ids) + [1] * len(output_ids),
            }
            file_obj.write(json.dumps(output, ensure_ascii=False) + "\n")
            file_obj.flush()


async def send_request(
    session, request_url, semaphore, file_lock, file_obj, payload, req_id, pbar
):
    start_time = time.perf_counter()
    try:
        payload = prepare_payload(payload)
        status = "success"
        async with semaphore:
            async with session.post(request_url, json=payload) as response:
                response_data = await response.json(content_type=None)
                input_ids, output_ids = None, None
                if response.status != 200:
                    logger.error(
                        f"HTTP {response.status}: "
                        f"{json.dumps(response_data, ensure_ascii=False)}"
                    )
                    status = "error"
                else:
                    input_ids, output_ids = extract_token_ids(response_data)
        result = {
            "req_id": req_id,
            "status": status,
            "latency": time.perf_counter() - start_time,
            "input_ids": input_ids,
            "output_ids": output_ids,
        }
    except Exception as error:
        result = {
            "req_id": req_id,
            "status": "error",
            "latency": time.perf_counter() - start_time,
            "record": payload,
            "error": str(error),
        }

    await write_result(file_obj, file_lock, result)
    pbar.update(1)
    return result


async def async_main(args):
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.resume_from < 0:
        raise ValueError("--resume-from must be >= 0")

    payloads = load_requests(args)
    indexed_payloads = select_payloads(
        payloads,
        resume_from=args.resume_from,
        num_requests=args.num_requests,
    )
    request_url = api_url(args.base_url, "/v1/chat/completions")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")

    semaphore = asyncio.Semaphore(args.max_concurrency)
    file_lock = asyncio.Lock()

    print(
        f"Starting {len(indexed_payloads)} requests from index "
        f"{args.resume_from} with concurrency "
        f"{args.max_concurrency}; output={args.output_file}",
        flush=True,
    )
    async with _create_bench_client_session(
        max_concurrency=args.max_concurrency, api_key=api_key
    ) as session:
        with (
            open(
                args.output_file,
                "a" if args.resume_from else "w",
                encoding="utf-8",
            ) as file_obj,
            tqdm(total=len(indexed_payloads), desc="Replying") as pbar,
        ):
            results = await asyncio.gather(
                *(
                    send_request(
                        session,
                        request_url,
                        semaphore,
                        file_lock,
                        file_obj,
                        payload,
                        req_id=req_id,
                        pbar=pbar,
                    )
                    for req_id, payload in indexed_payloads
                )
            )

    success_count = sum(result["status"] == "success" for result in results)
    print(f"Finished: {success_count}/{len(results)} succeeded", flush=True)
    return 0 if success_count == len(results) else 1


def main(argv: Optional[Sequence[str]] = None):
    return asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
