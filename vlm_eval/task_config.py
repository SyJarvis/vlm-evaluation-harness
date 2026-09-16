from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=None)
def load_task_config(name: str) -> dict[str, Any]:
    resource = files("vlm_eval").joinpath("config", f"{name}.json")
    with resource.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("name") != name:
        raise ValueError(f"task config {resource} does not declare name={name!r}")
    return config
