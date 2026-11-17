# 正确性评测

涵盖基准数据集准确率评测、思维链剥离与跨引擎 / 跨精度 Logits & Hidden States 数值对齐。

## 数据集评测 (`aib eval-dataset`)

::: code-group

```bash [GSM8K / GPQA / AIME25]
aib eval-dataset \
  --evals gsm8k \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --num-questions 50
```

```bash [HLE (本地数据集)]
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

| `--evals` | 数据集说明 | 额外依赖 |
| --- | --- | --- |
| `gsm8k` | 小学数学推理（自动剥离 `<think>` 与 `\boxed{}` / `####` 提取） | `ai-infra-bench[data]` |
| `gpqa` | GPQA Diamond 科学多项选择 | `ai-infra-bench[data]` |
| `aime25` | AIME 2025 竞赛数学 | `ai-infra-bench[data]` |
| `hle` | Humanity's Last Exam 纯文本子集 | `ai-infra-bench[data]` |
| `constrained_decoding` | Function Call 与 JSON Schema 结构化输出 | `ai-infra-bench[data]` |
| `deepswe` | 软件工程终端评测（Pier） | `ai-infra-bench[deepswe]` |

- 输出自动统计截断比例（`finish_reason == "length"`）；传入 `--repeat > 1` 时追加 `Multi-Round Summary`（均值、标准差与最值）。

## Logits 与 Hidden States 对齐

以第一个端点为基准，验证新量化方案、并行配置或推理后端的数值一致性：

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

- `eval-logits` 报告每个参考 token 位置的最大绝对误差、平均绝对误差（`mean |Δ|`）与通过率；`eval-hidden-states` 需服务端返回扩展 hidden states 字段。
