from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .api import OpenAIChatClient, OpenAIJudgeClient
from .evaluator import run_evaluation
from .task_config import load_task_config
from .tasks import ADAPTERS, DatasetConfig, get_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs through an OpenAI-compatible Chat API",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--tasks",
        help="comma-separated tasks; use 'all' or omit this option to run all tasks",
    )
    parser.add_argument(
        "--hub",
        choices=["hf", "modelscope", "local"],
        default="hf",
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--subset")
    parser.add_argument("--split")
    parser.add_argument("--revision", help="HF revision or ModelScope version")
    parser.add_argument("--cache-dir")
    parser.add_argument("--prompt-mode", choices=["direct", "cot"], default="direct")

    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--image-detail", choices=["auto", "low", "high"], default="auto")
    parser.add_argument("--extra-body", help="JSON object forwarded to the API request")

    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OPENAI_JUDGE_MODEL"),
        help="LLM judge model; CharXiv defaults to its official GPT-4o snapshot",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("OPENAI_JUDGE_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("OPENAI_JUDGE_API_KEY"),
    )
    parser.add_argument("--judge-timeout", type=float, default=300)

    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--rpm",
        type=int,
        help="cap request starts per minute to stay under provider rate limits",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tasks = _parse_tasks(args.tasks)
        _validate_multi_task_args(args, tasks)
    except ValueError as exc:
        parser.error(str(exc))
    extra_body = _parse_extra_body(args.extra_body)
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        image_detail=args.image_detail,
        extra_body=extra_body,
        rpm=args.rpm,
    )
    try:
        judge_client = _build_judge_client(args, tasks)
    except ValueError as exc:
        parser.error(str(exc))
    summaries = {}
    for task in tasks:
        adapter = get_adapter(task)
        dataset_config = DatasetConfig(
            source=args.hub,
            dataset_path=args.dataset_path or adapter.default_path(args.hub),
            subset=args.subset or adapter.default_subset,
            split=args.split or adapter.default_split,
            cache_dir=args.cache_dir,
            prompt_mode=args.prompt_mode,
            revision=args.revision,
        )
        summaries[task] = run_evaluation(
            adapter=adapter,
            dataset_config=dataset_config,
            client=client,
            judge_client=judge_client if task == "charxiv" else None,
            output_dir=_output_dir(args.output_dir, task, args.model, len(tasks)),
            concurrency=args.concurrency,
            limit=args.limit,
        )

    result = summaries[tasks[0]] if len(tasks) == 1 else summaries
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parse_extra_body(value: str | None) -> dict:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body must be a JSON object")
    return parsed


def _build_judge_client(args, tasks):
    if "charxiv" not in tasks:
        return None
    split = args.split or get_adapter("charxiv").default_split
    if split == "test":
        return None
    if not args.judge_api_key:
        raise ValueError(
            "CharXiv validation requires --judge-api-key or OPENAI_JUDGE_API_KEY"
        )
    judge = load_task_config("charxiv")["judge"]
    return OpenAIJudgeClient(
        model=args.judge_model or judge["model"],
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        temperature=judge["temperature"],
        top_p=judge["top_p"],
        seed=judge["seed"],
        response_format=judge["response_format"],
        timeout=args.judge_timeout,
    )


def _parse_tasks(value: str | None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(ADAPTERS)
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--tasks cannot be empty")
    if "all" in tasks:
        raise ValueError("'all' cannot be combined with explicit task names")
    unknown = [task for task in tasks if task not in ADAPTERS]
    if unknown:
        raise ValueError(f"unsupported tasks: {', '.join(unknown)}")
    return list(dict.fromkeys(tasks))


def _validate_multi_task_args(args: argparse.Namespace, tasks: list[str]) -> None:
    if len(tasks) == 1:
        return
    if args.hub != "hf":
        raise ValueError("multiple tasks currently require --hub hf")
    task_specific = [args.dataset_path, args.subset, args.split, args.revision]
    if any(value is not None for value in task_specific):
        raise ValueError(
            "--dataset-path, --subset, --split and --revision require exactly one task"
        )


def _output_dir(
    configured: str | None, task: str, model: str, task_count: int
) -> Path:
    if configured is not None:
        root = Path(configured)
        return root if task_count == 1 else root / task / _safe_name(model)
    return Path("outputs") / task / _safe_name(model)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


if __name__ == "__main__":
    main()
