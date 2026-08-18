# GSM8K Payload Dataset

`payload.jsonl` contains all 1,319 examples from `test.jsonl` converted to
OpenAI Chat Completions request bodies. Each line has this shape:

```json
{"messages":[{"role":"user","content":"Question: ...\nAnswer:"}]}
```

The source `answer` is intentionally excluded because this file is intended for
request benchmarking. The original `test.jsonl` remains available for
correctness evaluation.
