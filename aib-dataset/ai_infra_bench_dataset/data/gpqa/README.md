# GPQA Diamond Dataset

This directory contains the 198-question GPQA Diamond split published by the
GPQA authors and mirrored by OpenAI's simple-evals project:
<https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv>.

`gpqa_diamond.csv` is the source data and includes the answer labels for
correctness evaluation. `payload.jsonl` contains answer-free OpenAI Chat
Completions request bodies for `aib bench --dataset gpqa`; each question's four
choices are deterministically permuted when the payload file is built.
