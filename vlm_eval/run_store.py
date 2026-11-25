"""On-disk layout of one run: runs/<run_id>/{config.yaml,manifest.json,summary.json,tasks/}."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .run_config import config_sha256, dump_config, load_config
from .utils import atomic_write_json

TASK_STATUSES = ("pending", "generating", "scoring", "completed", "failed")


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.config_path = self.root / "config.yaml"
        self.manifest_path = self.root / "manifest.json"
        self.summary_path = self.root / "summary.json"

    @classmethod
    def create(
        cls, resolved: dict[str, Any], *, now: datetime | None = None
    ) -> "RunStore":
        now = now or datetime.now(timezone.utc)
        output_root = Path(resolved["run"]["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        base = f"{now:%Y%m%dT%H%M%SZ}__{safe_name(resolved['run']['name'])}"
        root = output_root / base
        attempt = 1
        while True:
            try:
                root.mkdir()
                break
            except FileExistsError:
                attempt += 1
                root = output_root / f"{base}_{attempt}"

        store = cls(root)
        store.config_path.write_text(dump_config(resolved), encoding="utf-8")
        timestamp = _isoformat(now)
        store._write_manifest(
            {
                "run_id": root.name,
                "created_at": timestamp,
                "updated_at": timestamp,
                "harness_version": __version__,
                "config_sha256": config_sha256(resolved),
                "model": resolved["model"]["name"],
                "tasks": {
                    task["id"]: {"name": task["name"], "status": "pending"}
                    for task in resolved["tasks"]
                },
            }
        )
        store.write_summary()
        return store

    @classmethod
    def open(cls, run_dir: str | Path) -> tuple["RunStore", dict[str, Any]]:
        store = cls(run_dir)
        if not store.manifest_path.exists() or not store.config_path.exists():
            raise FileNotFoundError(f"not a run directory: {store.root}")
        resolved = load_config(store.config_path)
        if config_sha256(resolved) != store.manifest()["config_sha256"]:
            raise ValueError(
                f"{store.config_path} no longer matches the hash recorded in "
                "manifest.json; start a new run instead of editing a finished one"
            )
        return store, resolved

    def manifest(self) -> dict[str, Any]:
        with self.manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def task_dir(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id

    def task_status(self, task_id: str) -> str:
        return self.manifest()["tasks"][task_id]["status"]

    def task_is_done(self, task_id: str) -> bool:
        """Completed with every sample answered; tasks with API errors are retried."""
        summary_path = self.task_dir(task_id) / "summary.json"
        if self.task_status(task_id) != "completed" or not summary_path.exists():
            return False
        with summary_path.open(encoding="utf-8") as handle:
            return json.load(handle)["num_errors"] == 0

    def start_task(self, task_id: str, context: dict[str, Any]) -> None:
        """Mark a task as generating and pin the settings it runs with.

        Context covers values the config hash cannot see, such as the bundled
        task protocol, so a resumed task never mixes two protocols.
        """
        context = json.loads(json.dumps(context, ensure_ascii=False))
        manifest = self.manifest()
        entry = manifest["tasks"][task_id]
        previous = entry.get("context")
        if previous is not None and previous != context:
            raise ValueError(
                f"task {task_id!r} settings differ from its earlier attempt; "
                "start a new run"
            )
        entry["context"] = context
        entry["status"] = "generating"
        entry.pop("error", None)
        self._write_manifest(manifest)
        self.write_summary()

    def set_task_status(
        self, task_id: str, status: str, *, error: str | None = None
    ) -> None:
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown task status: {status}")
        manifest = self.manifest()
        entry = manifest["tasks"][task_id]
        entry["status"] = status
        if error is not None:
            entry["error"] = error
        self._write_manifest(manifest)
        self.write_summary()

    def write_summary(self) -> dict[str, Any]:
        manifest = self.manifest()
        rows = []
        for task_id, entry in manifest["tasks"].items():
            row: dict[str, Any] = {
                "task_id": task_id,
                "task": entry["name"],
                "status": entry["status"],
            }
            summary_path = self.task_dir(task_id) / "summary.json"
            if entry["status"] == "completed" and summary_path.exists():
                with summary_path.open(encoding="utf-8") as handle:
                    row["metrics"] = json.load(handle)["metrics"]
            if entry.get("error"):
                row["error"] = entry["error"]
            rows.append(row)
        summary = {
            "run_id": manifest["run_id"],
            "model": manifest["model"],
            "tasks": rows,
        }
        atomic_write_json(self.summary_path, summary)
        return summary

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _isoformat(datetime.now(timezone.utc))
        atomic_write_json(self.manifest_path, manifest)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
