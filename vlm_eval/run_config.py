"""Experiment configuration: YAML in, fully resolved and secret-free dict out."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .task_config import load_task_config
from .tasks import ADAPTERS, get_adapter

HUBS = ("hf", "modelscope", "local")
PROMPT_MODES = ("direct", "cot")
IMAGE_DETAILS = ("auto", "low", "high")

GENERATION_DEFAULTS: dict[str, Any] = {
    "max_tokens": 128,
    "temperature": 0,
    "image_detail": "auto",
    "extra_body": {},
}
RUNTIME_DEFAULTS: dict[str, Any] = {"concurrency": 8, "rpm": None, "limit": None}

_TOP_KEYS = {"run", "model", "judge", "runtime", "tasks"}
_RUN_KEYS = {"name", "output_root"}
_MODEL_KEYS = {"name", "base_url", "api_key_env", "timeout", "max_retries"} | set(
    GENERATION_DEFAULTS
)
_JUDGE_KEYS = {"model", "base_url", "api_key_env", "timeout"}
_TASK_KEYS = {
    "id",
    "name",
    "hub",
    "dataset_path",
    "subset",
    "split",
    "revision",
    "prompt_mode",
    "generation",
    "runtime",
}
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    pass


def load_config(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return resolve_config(raw, limit=limit)


def resolve_config(raw: Any, *, limit: int | None = None) -> dict[str, Any]:
    """Fill defaults so the result fully describes the run.

    Resolving an already resolved config must return it unchanged; resume and
    the config hash depend on that.
    """
    root = _mapping(raw, "config", _TOP_KEYS)
    run = _mapping(root.get("run"), "run", _RUN_KEYS)
    model = _mapping(root.get("model"), "model", _MODEL_KEYS)
    judge = _mapping(root.get("judge"), "judge", _JUDGE_KEYS)
    runtime = _mapping(root.get("runtime"), "runtime", set(RUNTIME_DEFAULTS))
    if not model.get("name"):
        raise ConfigError("model.name is required")
    if limit is not None:
        _positive_int(limit, "--limit")

    generation = {
        **GENERATION_DEFAULTS,
        **{key: model[key] for key in GENERATION_DEFAULTS if key in model},
    }
    runtime = {**RUNTIME_DEFAULTS, **runtime}

    tasks_raw = root.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ConfigError("tasks must be a non-empty list")
    tasks = [
        _resolve_task(item, index, generation, runtime, limit)
        for index, item in enumerate(tasks_raw)
    ]
    ids = [task["id"] for task in tasks]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
    if duplicates:
        raise ConfigError(
            f"duplicate task ids: {', '.join(duplicates)}; set a distinct 'id' per task"
        )

    return {
        "run": {
            "name": str(run.get("name") or model["name"]),
            "output_root": str(run.get("output_root", "runs")),
        },
        "model": {
            "name": str(model["name"]),
            "base_url": model.get("base_url", "http://127.0.0.1:8000/v1"),
            "api_key_env": model.get("api_key_env", "OPENAI_API_KEY"),
            "timeout": model.get("timeout", 300),
            "max_retries": model.get("max_retries", 5),
        },
        "judge": {
            "model": judge.get("model", load_task_config("charxiv")["judge"]["model"]),
            "base_url": judge.get("base_url", "https://api.openai.com/v1"),
            "api_key_env": judge.get("api_key_env", "OPENAI_JUDGE_API_KEY"),
            "timeout": judge.get("timeout", 300),
        },
        "tasks": tasks,
    }


def config_sha256(resolved: dict[str, Any]) -> str:
    canonical = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dump_config(resolved: dict[str, Any]) -> str:
    return yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)


def needs_judge(task: dict[str, Any]) -> bool:
    return task["name"] == "charxiv" and task["split"] != "test"


def model_api_key(resolved: dict[str, Any]) -> str:
    return os.environ.get(resolved["model"]["api_key_env"]) or "local"


def judge_api_key(resolved: dict[str, Any]) -> str | None:
    if not any(needs_judge(task) for task in resolved["tasks"]):
        return None
    env = resolved["judge"]["api_key_env"]
    key = os.environ.get(env)
    if not key:
        raise ConfigError(f"CharXiv validation requires a judge API key in ${env}")
    return key


def _resolve_task(
    raw: Any,
    index: int,
    generation: dict[str, Any],
    runtime: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"name": raw}
    where = f"tasks[{index}]"
    item = _mapping(raw, where, _TASK_KEYS)
    name = item.get("name")
    if name not in ADAPTERS:
        raise ConfigError(
            f"{where}: unsupported task {name!r}; choose from {', '.join(ADAPTERS)}"
        )
    adapter = get_adapter(name)
    task_id = str(item.get("id") or name)
    if not _TASK_ID_RE.match(task_id) or task_id in {".", ".."}:
        raise ConfigError(f"{where}: id {task_id!r} must match {_TASK_ID_RE.pattern}")

    hub = item.get("hub", "hf")
    if hub not in HUBS:
        raise ConfigError(f"{where}: hub must be one of {', '.join(HUBS)}")
    dataset_path = item.get("dataset_path")
    if not dataset_path:
        if hub != "hf":
            raise ConfigError(f"{where}: dataset_path is required for hub {hub!r}")
        dataset_path = adapter.default_path(hub)
    prompt_mode = item.get("prompt_mode", "direct")
    if prompt_mode not in PROMPT_MODES:
        raise ConfigError(f"{where}: prompt_mode must be one of {', '.join(PROMPT_MODES)}")

    task_generation = copy.deepcopy(
        {
            **generation,
            **_mapping(
                item.get("generation"), f"{where}.generation", set(GENERATION_DEFAULTS)
            ),
        }
    )
    task_runtime = {
        **runtime,
        **_mapping(item.get("runtime"), f"{where}.runtime", set(RUNTIME_DEFAULTS)),
    }
    if limit is not None:
        task_runtime["limit"] = limit
    _validate_generation(task_generation, where)
    _validate_runtime(task_runtime, where)

    return {
        "id": task_id,
        "name": name,
        "hub": hub,
        "dataset_path": str(dataset_path),
        "subset": item.get("subset") or adapter.default_subset,
        "split": item.get("split") or adapter.default_split,
        "revision": item.get("revision"),
        "prompt_mode": prompt_mode,
        "generation": task_generation,
        "runtime": task_runtime,
    }


def _validate_generation(generation: dict[str, Any], where: str) -> None:
    _positive_int(generation["max_tokens"], f"{where}: max_tokens")
    if generation["image_detail"] not in IMAGE_DETAILS:
        raise ConfigError(f"{where}: image_detail must be one of {', '.join(IMAGE_DETAILS)}")
    if not isinstance(generation["extra_body"], dict):
        raise ConfigError(f"{where}: extra_body must be a mapping")


def _validate_runtime(runtime: dict[str, Any], where: str) -> None:
    _positive_int(runtime["concurrency"], f"{where}: concurrency")
    for key in ("rpm", "limit"):
        if runtime[key] is not None:
            _positive_int(runtime[key], f"{where}: {key}")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


def _mapping(value: Any, where: str, allowed: set[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {', '.join(unknown)}")
    return dict(value)
