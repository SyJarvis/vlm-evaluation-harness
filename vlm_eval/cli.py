from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .api import OpenAIChatClient, OpenAIJudgeClient
from .evaluator import run_evaluation
from .run_config import (
    ConfigError,
    judge_api_key,
    load_config,
    model_api_key,
    needs_judge,
)
from .run_store import RunStore
from .task_config import load_task_config
from .tasks import DatasetConfig, get_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-eval",
        description="Evaluate VLMs through an OpenAI-compatible Chat API",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start a new run from a YAML config")
    run.add_argument("config", help="path to the run config YAML")
    run.add_argument("--limit", type=int, help="evaluate only the first N samples per task")

    resume = commands.add_parser("resume", help="continue an existing run directory")
    resume.add_argument("run_dir", help="runs/<run_id> directory to resume")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        parser.error(
            "command-line evaluation options were removed; "
            "describe the run in YAML and use 'vlm-eval run <config.yaml>'"
        )
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            resolved = load_config(args.config, limit=args.limit)
            api_key, judge_key = model_api_key(resolved), judge_api_key(resolved)
            store = RunStore.create(resolved)
        else:
            store, resolved = RunStore.open(args.run_dir)
            api_key, judge_key = model_api_key(resolved), judge_api_key(resolved)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"run directory: {store.root}")
    summary = execute_run(store, resolved, api_key=api_key, judge_key=judge_key)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def execute_run(
    store: RunStore,
    resolved: dict[str, Any],
    *,
    api_key: str,
    judge_key: str | None,
) -> dict[str, Any]:
    judge_client = (
        _build_judge_client(resolved["judge"], judge_key)
        if judge_key is not None
        else None
    )
    for task in resolved["tasks"]:
        task_id = task["id"]
        if store.task_is_done(task_id):
            continue
        adapter = get_adapter(task["name"])
        client = _build_chat_client(resolved["model"], task, api_key)
        task_judge = judge_client if needs_judge(task) else None
        dataset_config = DatasetConfig(
            source=task["hub"],
            dataset_path=task["dataset_path"],
            subset=task["subset"],
            split=task["split"],
            prompt_mode=task["prompt_mode"],
            revision=task["revision"],
        )
        context: dict[str, Any] = {"api": client.public_config()}
        if adapter.protocol_config():
            context["task_protocol"] = adapter.protocol_config()
        if task_judge is not None:
            context["judge_api"] = task_judge.public_config()

        store.start_task(task_id, context)
        try:
            run_evaluation(
                adapter=adapter,
                dataset_config=dataset_config,
                client=client,
                judge_client=task_judge,
                output_dir=store.task_dir(task_id),
                concurrency=task["runtime"]["concurrency"],
                limit=task["runtime"]["limit"],
                task_id=task_id,
                on_stage=lambda stage, task_id=task_id: store.set_task_status(task_id, stage),
            )
        except BaseException as exc:
            store.set_task_status(task_id, "failed", error=f"{type(exc).__name__}: {exc}")
            raise
        store.set_task_status(task_id, "completed")
    return store.write_summary()


def _build_chat_client(model: dict, task: dict, api_key: str) -> OpenAIChatClient:
    generation = task["generation"]
    return OpenAIChatClient(
        model=model["name"],
        base_url=model["base_url"],
        api_key=api_key,
        max_tokens=generation["max_tokens"],
        temperature=generation["temperature"],
        timeout=model["timeout"],
        max_retries=model["max_retries"],
        image_detail=generation["image_detail"],
        extra_body=generation["extra_body"],
        rpm=task["runtime"]["rpm"],
    )


def _build_judge_client(judge: dict, api_key: str) -> OpenAIJudgeClient:
    protocol = load_task_config("charxiv")["judge"]
    return OpenAIJudgeClient(
        model=judge["model"],
        base_url=judge["base_url"],
        api_key=api_key,
        temperature=protocol["temperature"],
        top_p=protocol["top_p"],
        seed=protocol["seed"],
        response_format=protocol["response_format"],
        timeout=judge["timeout"],
    )


if __name__ == "__main__":
    main()
