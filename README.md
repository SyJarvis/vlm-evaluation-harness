# VLM Evaluation Harness

通过 OpenAI-compatible Chat Completions API 客观评测视觉语言模型。当前支持：

- MMMU
- MMMU-Pro：`standard (10 options)`、`standard (4 options)`、`vision`
- MMStar
- CharXiv：descriptive + reasoning，使用官方 GPT-4o judge 协议
- ChartQA：支持 Relaxed Accuracy（5% 数值容差）及 human / augmented 细分统计
- DocVQA：支持 ANLS（Average Normalized Levenshtein Similarity）评测
- BabyVision (`UnipatAI/BabyVision`)：涵盖细粒度辨别、空间感知、视觉追踪、视觉模式识别（支持坐标提取与选择题评测）
- MMLU (`cais/mmlu`)：语言理解与多学科通识基准

工具只调用 API，不在评测进程中加载模型。数据可来自 Hugging Face、ModelScope 或本地路径。

## 安装

```bash
cd /mnt/data3/runke.zhong/research/projects/vlm-evaluation-harness
pip install -e .

# 需要 ModelScope 时
pip install -e '.[modelscope]'
```

使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

```bash
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=local

vlm-eval \
  --tasks mmstar \
  --hub hf \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1 \
  --concurrency 8 \
  --output-dir outputs/mmstar/qwen3.5-vl
```

MMMU validation：

```bash
vlm-eval \
  --tasks mmmu \
  --hub hf \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1 \
  --concurrency 8
```

MMMU-Pro 默认使用官方 `standard (10 options)` direct setting：

```bash
vlm-eval \
  --tasks mmmu_pro \
  --hub hf \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1

# vision-only + CoT
vlm-eval \
  --tasks mmmu_pro \
  --hub hf \
  --subset vision \
  --prompt-mode cot \
  --model glm-5.3-flash
  --base-url http://127.0.0.1:18080/v1

vlm-eval \
  --tasks babyvision \
  --hub hf \
  --model glm-5.3-flash \
  --max-tokens 4096 \
  --concurrency 8 \
  --output-dir outputs/babyvision/glm-5.3-flash
  --base-url http://127.0.0.1:18080/v1
```

CharXiv validation 需要一把独立的 judge key。被测模型和 judge 可以使用不同的 OpenAI-compatible 服务：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=local
export OPENAI_JUDGE_API_KEY=<openai-key>

vlm-eval \
  --tasks charxiv \
  --hub hf \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1
```

CharXiv judge 默认配置为官方的 `gpt-4o-2024-05-13` 和 `https://api.openai.com/v1`。也可以显式设置：

```bash
vlm-eval \
  --tasks charxiv \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1 \
  --judge-model gpt-4o-2024-05-13 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-api-key "$OPENAI_JUDGE_API_KEY"
```

使用其他 judge 模型仍可运行，但结果会标记为非官方可比。public test 的答案全部隐藏，因此 `--split test` 只生成响应，不执行本地评分。

## 数据源

Hugging Face 使用每个 benchmark 的官方仓库作为默认路径。`datasets` 会自动读取 `HF_ENDPOINT`。

```bash
vlm-eval --tasks mmstar --hub hf ...
```

ModelScope 和本地数据要求显式指定路径：

```bash
vlm-eval \
  --tasks mmmu \
  --hub modelscope \
  --dataset-path AI-ModelScope/MMMU \
  ...

vlm-eval \
  --tasks mmstar \
  --hub local \
  --dataset-path /data/MMStar \
  ...
```

CharXiv 还支持直接指定已下载的官方源码目录：

```bash
vlm-eval \
  --tasks charxiv \
  --hub local \
  --dataset-path /home/runke.zhong/research/projects/vlm-evaluation-harness/lib/charxiv \
  --split validation \
  --model qwen3.5-vl \
  --base-url http://127.0.0.1:18080/v1
```

该目录需要包含官方 `data/` JSON 和已下载、解压后的 `images/*.jpg`。

本地路径支持 Hugging Face `save_to_disk` 目录、官方数据集仓库目录，以及 JSON/JSONL、CSV/TSV、Parquet、Arrow 文件。

正式对比建议用 `--revision <commit-or-tag>` 固定 Hugging Face revision 或 ModelScope version；该值会写入 `run_config.json`。

多个任务用逗号分隔；`--tasks all` 或不指定 `--tasks` 都会运行全部任务：

```bash
vlm-eval --model qwen3.5-vl --tasks mmstar,mmmu --hub hf ...
vlm-eval --model qwen3.5-vl --tasks all --hub hf ...
vlm-eval --model qwen3.5-vl --hub hf ...
```

由于 `all` 包含 CharXiv validation，上述后两种命令需要设置 `OPENAI_JUDGE_API_KEY`。

多任务模式下，`--output-dir` 被视为输出根目录，每个任务写入独立的 `<output-dir>/<task>/<model>/`。`--dataset-path`、`--subset`、`--split` 和 `--revision` 只允许用于单任务运行。

## 输出与恢复

每个输出目录包含：

- `run_config.json`：不含 API key 的运行配置
- `responses.jsonl`：原始 API 响应，逐条追加，可断点恢复
- `predictions.jsonl`：官方口径解析后的答案和逐题正确性
- `summary.json`：总分和分类统计
- `judge_calls.jsonl`：CharXiv judge 请求与 JSON 结果，可断点恢复

重复执行相同命令会跳过已有成功响应并重试失败项。为避免混淆，不允许在同一输出目录中混用不同模型、数据集或提示词配置。

提示词与评分口径见 [docs/datasets.md](docs/datasets.md)。
