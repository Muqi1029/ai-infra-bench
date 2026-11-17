# Evaluation

Benchmark dataset accuracy with automatic thinking-block stripping, or verify numerical alignment across engines and precisions.

## Dataset Evaluation (`aib eval-dataset`)

::: code-group

```bash [GSM8K / GPQA / AIME25]
aib eval-dataset \
  --evals gsm8k \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --num-questions 50
```

```bash [HLE (Local)]
aib eval-dataset \
  --evals hle \
  --dataset-path ./hle.jsonl \
  --base-url http://127.0.0.1:30000 \
  --model your-model
```

```bash [Constrained / DeepSWE]
aib eval-dataset --evals constrained_decoding --base-url http://127.0.0.1:30000 --model your-model
aib eval-dataset --evals deepswe --base-url http://127.0.0.1:30000 --model your-model
```

:::

| `--evals` | Scope | Extra |
| --- | --- | --- |
| `gsm8k` | Grade-school math (strips `<think>`, parses `\boxed{}` & `####`) | `ai-infra-bench[data]` |
| `gpqa` | GPQA Diamond scientific multiple-choice | `ai-infra-bench[data]` |
| `aime25` | AIME 2025 competition math | `ai-infra-bench[data]` |
| `hle` | Humanity's Last Exam text subset | `ai-infra-bench[data]` |
| `constrained_decoding` | Function-calling & JSON Schema adherence | `ai-infra-bench[data]` |
| `deepswe` | Software engineering harness via Pier | `ai-infra-bench[deepswe]` |

- Reports truncation rate (`finish_reason == "length"`) and adds `Multi-Round Summary` (mean, std, min/max) when `--repeat > 1`.

## Logits & Hidden States Alignment

Compare candidate endpoints against the first reference URL to catch quantization or backend drift:

::: code-group

```bash [eval-logits]
aib eval-logits \
  --base-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --model your-model \
  --prompt "The capital of France is" \
  --atol 0.01
```

```bash [eval-hidden-states]
aib eval-hidden-states \
  --base-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --model your-model \
  --prompt "Hello" \
  --atol 0.01
```

:::

- `eval-logits` reports max abs error, mean absolute error (`mean |Δ|`), and pass rate over reference Top-K tokens; `eval-hidden-states` requires hidden states in responses.
