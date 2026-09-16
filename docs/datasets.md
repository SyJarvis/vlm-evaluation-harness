# Supported datasets

## MMMU

- HF repository: `MMMU/MMMU`
- Default split: `validation`
- Default subset: all 30 subjects
- Multiple-choice prompt:

  ```text
  {question}
  (A) ...
  (B) ...

  Answer with the option's letter from the given choices directly.
  ```

- Short-answer prompt ends with `Answer the question using a single word or phrase.`
- Multiple-choice parsing and open-answer normalization follow the official MMMU implementation.
- Images referenced by `<image N>` are sent at the corresponding position and mapped to `image_N`.

## MMMU-Pro

- HF repository: `MMMU/MMMU_Pro`
- Default split: `test`
- Default setting: `standard (10 options)`
- Supported settings: `standard (10 options)`, `standard (4 options)`, `vision`
- Supported prompt modes: `direct` and `cot`
- The exact official prompt suffixes are defined in `vlm_eval/prompts.py`.
- Standard-setting image order follows the `<image N>` references rather than the numeric order of dataset columns.

## MMStar

- HF repository: `Lin-Chen/MMStar`
- Default config/split: `val` / `val`
- The dataset question already contains its choices. The evaluator appends only the direct-answer letter instruction.
- Metrics include overall accuracy, six coarse capability groups and 18 fine-grained groups.

## Alignment references

- MMMU and MMMU-Pro: `MMMU-Benchmark/MMMU` commit `268471d0d488258990025331c7528359c324aa25`
- MMStar: `MMStar-Benchmark/MMStar` commit `88f243ab4a39cb339530085c33aecb22819881a1`

The default prompts intentionally add no system message. Changing prompt mode, adding server-side chat-template behavior, or injecting an API-specific system prompt can make scores incomparable.


## princeton-nlp/CharXiv

- HF repository: `princeton-nlp/CharXiv`
- Default split: `validation`
- Each chart expands into four descriptive questions and one reasoning question.
- Validation contains 1,000 charts: 4,000 descriptive queries and 1,000 reasoning queries.
- Public test contains 1,323 charts, but all answers are hidden and cannot be scored locally.
- Requests follow the official content order: question text first, then the chart image.
- Generation uses `max_tokens=1000`, `temperature=0`, `top_p=1`, and `seed=42`.

CharXiv has two separate leaderboard metrics. The evaluator reports descriptive and reasoning scores independently and does not create a synthetic combined score.

Scoring is a second API stage using the official judge protocol:

- Official judge: `gpt-4o-2024-05-13`
- JSON response format
- `temperature=0`, `top_p=1`, `seed=42`
- Descriptive questions are grouped by qid and graded in batches of five.
- Reasoning questions use one judge request per chart.
- Invalid judge outputs receive score `-1` and count as zero during aggregation.
- A complete validation run uses 807 descriptive judge calls and 1,000 reasoning judge calls.

The evaluated model key is loaded from `--api-key` or `OPENAI_API_KEY`. The independent judge key is loaded from `--judge-api-key` or `OPENAI_JUDGE_API_KEY`; keys are never written to run artifacts.

The public test split has an upstream limitation: Number-in-General prompts normally derive the required decimal precision from the hidden reference answer. Since that answer is unavailable, the evaluator omits this precision suffix and records `precision_instruction_omitted` in sample metadata.

Official prompts, rubrics, mappings and pinned upstream revision are stored in `vlm_eval/config/charxiv.json`. The implementation is aligned with `princeton-nlp/CharXiv` commit `7ebe88f78dee387691551f071abcb2b9e1a8025b`.

## UnipatAI/BabyVision

- HF repository: `UnipatAI/BabyVision`
- Default config/split: `default` / `train` (the only split, 388 questions)
- Two answer types: `choice` (options formatted as letters A–F, gold answer converted from the 1-indexed `choiceAns` field) and `blank` (free-form short answer).
- Blank answers may be grid coordinates; the evaluator matches coordinates like `(x, y)` tolerantly, including full-width punctuation (`，`、`（）`), and falls back to normalized open-answer matching.
- Metrics: overall accuracy grouped by `type` (4 coarse abilities), `subtype`, and `ans_type`.

## MMLU

- HF repository: `cais/mmlu`
- Default config/split: `all` / `test` (14,042 questions across 57 subjects)
- Four-option multiple choice, text-only (no image): the prompt lists the question and options `(A)`–`(D)` followed by the direct-answer letter instruction.
- Metrics: overall accuracy and per-subject grouping (`subject`).

## ChartQA

- HF repository: `HuggingFaceM4/ChartQA`
- Default split: `test`
- Evaluation metric: **Relaxed Accuracy** (relative error $\le 5\%$ for numbers, exact string match for text).
- Reports overall accuracy, relaxed accuracy, and splits by type (`human`, `augmented`).

## DocVQA

- HF repository: `lmms-lab/DocVQA`
- Default config / split: `DocVQA` / `validation`
- Evaluation metric: **ANLS** (Average Normalized Levenshtein Similarity, threshold $\tau = 0.5$).
- Reports average ANLS and binary accuracy (ANLS $\ge 0.5$).
