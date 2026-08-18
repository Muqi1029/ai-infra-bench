import json
from argparse import Namespace
from pathlib import Path

from scripts.build_payload_datasets import build_gsm8k_payloads, build_sharegpt_payloads


def test_build_gsm8k_payloads(tmp_path):
    source_path = tmp_path / "test.jsonl"
    source_path.write_text(
        json.dumps({"question": "What is 1 + 1?", "answer": "#### 2"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "gsm8k" / "payload.jsonl"

    assert build_gsm8k_payloads(source_path, output_path) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "messages": [{"role": "user", "content": "Question: What is 1 + 1?\nAnswer:"}]
    }


def test_build_sharegpt_payloads(tmp_path):
    source_path = tmp_path / "sharegpt.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "id": "valid",
                    "conversations": [
                        {"from": "human", "value": "Hello"},
                        {"from": "gpt", "value": "Hi"},
                    ],
                },
                {
                    "id": "missing-assistant",
                    "conversations": [{"from": "human", "value": "Skip me"}],
                },
                {
                    "id": "private-key",
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "-----BEGIN " + "PRIVATE KEY-----\nsecret\n"
                                "-----END PRIVATE KEY-----"
                            ),
                        },
                        {"from": "gpt", "value": "Do not publish it"},
                    ],
                },
                {
                    "id": "valid-second",
                    "conversations": [
                        {"from": "human", "value": "World"},
                        {"from": "gpt", "value": "Hello"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "sharegpt" / "payload.jsonl"

    assert build_sharegpt_payloads(source_path, output_path, max_shard_bytes=60) == (
        2,
        2,
    )
    shard_paths = sorted(output_path.parent.glob("payload-*.jsonl"))
    assert [
        json.loads(shard_path.read_text(encoding="utf-8")) for shard_path in shard_paths
    ] == [
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "World"}]},
    ]
    assert all(shard_path.stat().st_size <= 60 for shard_path in shard_paths)


def test_benchmark_loads_packaged_payload_shards(tmp_path, monkeypatch):
    from ai_infra_bench.performance import bench

    data_dir = tmp_path / "data" / "sharegpt"
    data_dir.mkdir(parents=True)
    (data_dir / "payload.jsonl").write_text('{"legacy":true}\n', encoding="utf-8")
    (data_dir / "payload-00001.jsonl").write_text(
        '{"messages":[{"role":"user","content":"second"}]}\n',
        encoding="utf-8",
    )
    (data_dir / "payload-00000.jsonl").write_text(
        '{"messages":[{"role":"user","content":"first"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(bench.resources, "files", lambda _: tmp_path)

    requests = bench.read_packaged_requests("sharegpt")

    assert [request["messages"][0]["content"] for request in requests] == [
        "first",
        "second",
    ]


def test_benchmark_loads_packaged_gsm8k_payloads(monkeypatch):
    package_parent = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(package_parent))

    from ai_infra_bench.performance.bench import load_requests

    requests = load_requests(
        Namespace(
            dataset="gsm8k",
            filter_constrained_grammar_requests=False,
        )
    )

    assert len(requests) == 1319
    assert requests[0]["messages"][0]["role"] == "user"
