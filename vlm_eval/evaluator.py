"""Evaluation orchestration for a single task directory."""

from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from .sample import EvalSample
from .tasks.base import DatasetAdapter, DatasetConfig
from .utils import atomic_write_json, atomic_write_jsonl, load_latest_jsonl

RESPONSE_FIELDS = ("response", "error", "latency_seconds", "request_options")


def run_evaluation(
    *,
    adapter: DatasetAdapter,
    dataset_config: DatasetConfig,
    client: Any,
    judge_client: Any | None = None,
    output_dir: str | Path,
    concurrency: int = 8,
    limit: int | None = None,
    task_id: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_path = output_path / "samples.jsonl"
    responses_path = output_path / "responses.jsonl"

    stored_samples = _read_jsonl(samples_path) if samples_path.exists() else None
    completed_ids = _completed_ids(load_latest_jsonl(responses_path))
    if stored_samples is not None and all(
        sample["id"] in completed_ids for sample in stored_samples
    ):
        samples = stored_samples
    else:
        samples = _generate(
            adapter=adapter,
            dataset_config=dataset_config,
            client=client,
            samples_path=samples_path,
            responses_path=responses_path,
            stored_samples=stored_samples,
            completed_ids=completed_ids,
            concurrency=concurrency,
            limit=limit,
        )

    if on_stage is not None:
        on_stage("scoring")
    latest = load_latest_jsonl(responses_path)
    missing = [sample["id"] for sample in samples if sample["id"] not in latest]
    if missing:
        raise RuntimeError(f"missing API records for {len(missing)} samples")

    records = [
        {
            **sample,
            **{field: latest[sample["id"]].get(field) for field in RESPONSE_FIELDS},
        }
        for sample in samples
    ]
    evaluated, raw_summary = adapter.evaluate_with_context(
        records,
        judge_client=judge_client,
        output_dir=output_path,
    )
    summary = build_task_summary(task_id or adapter.name, adapter.name, raw_summary, evaluated)
    atomic_write_jsonl(output_path / "predictions.jsonl", evaluated)
    atomic_write_json(output_path / "summary.json", summary)
    return summary


def build_task_summary(
    task_id: str,
    dataset: str,
    raw: dict[str, Any],
    evaluated: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap an adapter summary so every task exposes `metrics` and `groups`."""
    summary: dict[str, Any] = {
        "task_id": task_id,
        "dataset": dataset,
        "status": "completed",
        "num_samples": len(evaluated),
        "num_errors": sum(bool(item.get("error")) for item in evaluated),
        "metrics": {},
        "groups": {},
    }
    for key, value in raw.items():
        if key in ("dataset", "num_samples"):
            continue
        if key.startswith("by_"):
            summary["groups"][key[len("by_") :]] = value
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not key.startswith("num_")
        ):
            summary["metrics"][key] = value
        else:
            summary[key] = value
    for mode in ("descriptive", "reasoning"):
        stats = raw.get(mode)
        if isinstance(stats, dict) and "Overall Score" in stats:
            summary["metrics"][f"{mode}_overall"] = stats["Overall Score"]
    return summary


def _generate(
    *,
    adapter: DatasetAdapter,
    dataset_config: DatasetConfig,
    client: Any,
    samples_path: Path,
    responses_path: Path,
    stored_samples: list[dict[str, Any]] | None,
    completed_ids: set[str],
    concurrency: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    pending: dict[Future, tuple[EvalSample, int]] = {}
    max_pending = max(concurrency * 2, 1)

    with responses_path.open("a", encoding="utf-8") as response_file:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            progress = tqdm(desc=f"{adapter.name} API evaluation", unit="sample")
            try:
                for sequence, row in enumerate(adapter.iter_rows(dataset_config)):
                    if limit is not None and sequence >= limit:
                        break
                    sample = adapter.build_sample(row, dataset_config)
                    if sample.id in seen_ids:
                        raise ValueError(f"duplicate sample id: {sample.id}")
                    seen_ids.add(sample.id)
                    snapshot = _sample_snapshot(sample, sequence)
                    if stored_samples is not None and (
                        sequence >= len(stored_samples)
                        or stored_samples[sequence] != snapshot
                    ):
                        raise ValueError(
                            f"dataset no longer matches {samples_path} at sample "
                            f"{sample.id!r}; pin a revision or start a new run"
                        )
                    samples.append(snapshot)
                    if sample.id in completed_ids:
                        progress.update(1)
                        continue
                    future = executor.submit(_call_api, client, sample, sequence)
                    pending[future] = (sample, sequence)
                    if len(pending) >= max_pending:
                        _drain_one(pending, response_file, progress)

                if stored_samples is None:
                    atomic_write_jsonl(samples_path, samples)
                elif len(samples) != len(stored_samples):
                    raise ValueError(
                        f"dataset yielded {len(samples)} samples but {samples_path} "
                        f"has {len(stored_samples)}; start a new run"
                    )
                while pending:
                    _drain_one(pending, response_file, progress)
            finally:
                progress.close()
    return samples


def _sample_snapshot(sample: EvalSample, sequence: int) -> dict[str, Any]:
    snapshot = {
        "id": sample.id,
        "sequence": sequence,
        "prompt": sample.audit_prompt(),
        "answer": sample.answer,
        "question_type": sample.question_type,
        "choices": sample.choices,
        "index_to_answer": sample.index_to_answer,
        "metadata": sample.metadata,
    }
    return json.loads(json.dumps(snapshot, ensure_ascii=False))


def _call_api(client: Any, sample: EvalSample, sequence: int) -> dict[str, Any]:
    started = time.monotonic()
    response = None
    error = None
    try:
        response = client.complete(sample)
    except Exception as exc:  # Persist errors so a later run can retry them.
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": sample.id,
        "sequence": sequence,
        "response": response,
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 6),
        "request_options": sample.request_options,
    }


def _drain_one(
    pending: dict[Future, tuple[EvalSample, int]],
    response_file,
    progress: tqdm,
) -> None:
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        pending.pop(future)
        record = future.result()
        response_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        response_file.flush()
        progress.update(1)


def _completed_ids(latest: dict[str, dict[str, Any]]) -> set[str]:
    return {
        sample_id
        for sample_id, record in latest.items()
        if not record.get("error") and record.get("response") is not None
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
