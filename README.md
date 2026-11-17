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
cd vlm-evaluation-harness
pip install -e .

# 需要 ModelScope 时
pip install -e '.[modelscope]'
```

使用 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

一次评测由一份 YAML 配置描述。最小配置：

```yaml
model:
  name: qwen3.5-vl
  base_url: http://127.0.0.1:18080/v1
tasks:
  - mmstar
```

```bash
export OPENAI_API_KEY=local
vlm-eval run configs/mmstar.yaml

# 冒烟测试：每个任务只跑前 N 条
vlm-eval run configs/mmstar.yaml --limit 20
```

每次 `run` 都会新建一个 `runs/<UTC时间>__<run.name>/` 目录，命令开头会打印它的路径，结束时打印套件汇总。完整示例见 [configs/example.yaml](configs/example.yaml)。

## 配置

```yaml
run:
  name: qwen3.5-vl-suite        # 目录名后缀，默认取 model.name
  output_root: runs             # 默认 runs

model:
  name: qwen3.5-vl
  base_url: http://127.0.0.1:18080/v1
  api_key_env: OPENAI_API_KEY   # 只写环境变量名
  timeout: 300
  max_retries: 5
  max_tokens: 128               # 以下四项是生成参数的全局默认值
  temperature: 0
  image_detail: auto            # auto / low / high
  extra_body: {}                # 原样转发给 API 的字段

judge:                          # 仅 CharXiv validation 使用
  model: gpt-4o-2024-05-13
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_JUDGE_API_KEY

runtime:
  concurrency: 8
  rpm: null                     # 每分钟最多发起的请求数
  limit: null

tasks:
  - mmstar                      # 只写任务名时全部使用默认值
  - id: mmmu_pro_vision_cot     # 产物目录名，默认等于 name，同一 run 内不能重复
    name: mmmu_pro
    hub: hf                     # hf / modelscope / local
    subset: vision
    split: test
    revision: null              # 正式对比建议固定 commit 或 tag
    prompt_mode: cot            # direct / cot
    generation:
      max_tokens: 2048
    runtime:
      concurrency: 4
```

任务项里的 `generation` 和 `runtime` 覆盖全局默认值，所以同一个 run 可以给不同任务设置不同的 `max_tokens`、并发和限速。`dataset_path`、`subset`、`split` 省略时使用各任务的官方默认值，见 [docs/datasets.md](docs/datasets.md)。配置中出现未知字段会直接报错。

密钥只通过环境变量传入，`config.yaml` 和其他产物里只出现变量名。`model.api_key_env` 对应的变量未设置时使用 `local`，适合本地 vLLM 等服务。

### CharXiv

CharXiv validation 需要一把独立的 judge key，被测模型和 judge 可以使用不同的 OpenAI-compatible 服务：

```bash
export OPENAI_JUDGE_API_KEY=<openai-key>
```

judge 默认使用官方的 `gpt-4o-2024-05-13` 和 `https://api.openai.com/v1`。配置里包含 CharXiv validation 任务但缺少 judge key 时，`run` 会在创建目录和发出请求之前报错。使用其他 judge 模型仍可运行，但结果会标记为非官方可比。public test 的答案全部隐藏，因此 `split: test` 只生成响应，不执行本地评分，也不需要 judge key。

CharXiv 的官方生成参数（如 `max_tokens=1000`）由任务协议固定，会覆盖配置里的 `generation`。

### 数据源

Hugging Face 使用每个 benchmark 的官方仓库作为默认路径，`datasets` 会自动读取 `HF_ENDPOINT`。ModelScope 和本地数据要求显式指定 `dataset_path`：

```yaml
tasks:
  - name: mmmu
    hub: modelscope
    dataset_path: AI-ModelScope/MMMU
  - name: mmstar
    hub: local
    dataset_path: /data/MMStar
  - name: charxiv
    hub: local
    dataset_path: /path/to/charxiv   # 官方源码目录，含 data/ JSON 和解压后的 images/*.jpg
    split: validation
```

本地路径支持 Hugging Face `save_to_disk` 目录、官方数据集仓库目录，以及 JSON/JSONL、CSV/TSV、Parquet、Arrow 文件。

### 速率控制

商用 API 通常有账号级限速。被 429 限流时客户端会指数退避重试（8s 起步、上限 60s、尊重 `Retry-After`，至多 6 次）。也可以用 `runtime.rpm` 主动限制每分钟发起的请求数：

```yaml
tasks:
  - name: babyvision
    generation:
      max_tokens: 8192
    runtime:
      concurrency: 4
      rpm: 30
```

## 输出与恢复

```text
runs/20261026T011130Z__qwen3.5-vl-suite/
  config.yaml          # 补全默认值后的实际配置，不含密钥
  manifest.json        # 配置哈希、harness 版本、各任务状态与实际 API 参数
  summary.json         # 套件汇总：每个任务的状态和主指标
  tasks/<task_id>/
    samples.jsonl      # 题目快照：prompt、标准答案、选项、metadata
    responses.jsonl    # 原始 API 响应，逐条追加
    predictions.jsonl  # 解析后的答案和逐题正确性，附带题目和响应
    summary.json       # 该任务的 metrics、groups 和任务特有统计
    judge_calls.jsonl  # 仅 CharXiv：judge 请求与 JSON 结果
```

任务状态依次为 `pending`、`generating`、`scoring`、`completed`；出错时为 `failed`，并在 `manifest.json` 和 `summary.json` 中记录错误信息。

任务 `summary.json` 统一包含 `num_samples`、`num_errors`、`metrics`（如 `accuracy`、`anls`、`relaxed_accuracy`）和 `groups`（如按 subject、category 的分组统计）。CharXiv 在 `metrics` 中分别给出 `descriptive_overall` 和 `reasoning_overall`，不合成总分；完整官方统计保留在 `descriptive`、`reasoning` 字段中。

中断或部分请求失败后，用 `resume` 继续同一个目录：

```bash
vlm-eval resume runs/20261026T011130Z__qwen3.5-vl-suite
```

`resume` 读取目录里的 `config.yaml`，如果它与 `manifest.json` 里的哈希不一致就拒绝运行；要改配置，请新建一个 run。已完成且没有 API 错误的任务会被跳过，其余任务只重发缺失或失败的请求，然后重新评分；CharXiv judge 按 `judge_calls.jsonl` 断点继续。全部响应成功时，评分直接使用 `samples.jsonl`，不再读取数据集；仍有请求需要重发时会重新读取数据集，题目与 `samples.jsonl` 不一致时报错，因此建议固定 `revision`。

提示词与评分口径见 [docs/datasets.md](docs/datasets.md)。
