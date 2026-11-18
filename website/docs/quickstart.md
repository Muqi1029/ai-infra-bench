# 快速开始

需 Python 3.10+ 与运行中的 OpenAI 兼容推理服务（默认 `http://127.0.0.1:30000`）。

## 1. 安装

::: code-group

```bash [pip]
pip install ai-infra-bench
# 可选：数据集评测与分词 / 权重参数统计
pip install 'ai-infra-bench[data,weight]'
```

```bash [uv]
uv tool install ai-infra-bench
# 或在当前虚拟环境中安装扩展包
uv pip install 'ai-infra-bench[data,weight]'
```

:::

## 2. 验证连通性

发送单条流式请求，确认接口与认证正常：

```bash
aib req \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --prompt "Explain prefix caching in one sentence."
```

## 3. 压测并生成报告

运行随机 token 压测并导出交互式 HTML 图表：

```bash
aib bench \
  --base-url http://127.0.0.1:30000 \
  --model your-model \
  --dataset random \
  --input-len 1024 --output-len 256 \
  --num-requests 100 \
  --max-concurrency 1 8 32 \
  --goodput ttft:500 tpot:50 \
  --disable-flush-cache \
  --metric-path metrics.yaml

aib plot-metrics metrics.yaml -o metrics.html
```

## 下一步

- [压测与 SLO](/benchmark)：随机负载、前缀缓存、多轮会话回放与 SLO 二分搜索
- [正确性评测](/evaluation)：数据集评测、Logits 与 Hidden States 对齐
- [请求与诊断工具](/tools)：管道请求调试、监控看板与 GPU / 权重检查
- [命令与指标参考](/reference)：全部 14 个子命令、指标公式与常见问题
