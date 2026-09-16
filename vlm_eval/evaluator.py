"""Evaluation orchestration."""

from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .sample import EvalSample
from .tasks.base import DatasetAdapter, DatasetConfig
from .utils import atomic_write_json, atomic_write_jsonl, load_latest_jsonl


def run_evaluation(
    *,
    adapter: DatasetAdapter,
    dataset_config: DatasetConfig,
    client: Any,
    judge_client: Any | None = None,
    output_dir: str | Path,
    concurrency: int = 8,
    limit: int | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    responses_path = output_path / "responses.jsonl"
    predictions_path = output_path / "predictions.jsonl"
    summary_path = output_path / "summary.json"
    run_config_path = output_path / "run_config.json"

    run_config = {
        "dataset": adapter.name,
        "source": dataset_config.source,
        "dataset_path": dataset_config.dataset_path,
        "subset": dataset_config.subset,
        "split": dataset_config.split,
        "revision": dataset_config.revision,
        "prompt_mode": dataset_config.prompt_mode,
        "limit": limit,
        "api": client.public_config(),
    }
    task_protocol = adapter.protocol_config()
    if task_protocol:
        run_config["task_protocol"] = task_protocol
    if judge_client is not None:
        run_config["judge_api"] = judge_client.public_config()
    _validate_or_write_config(run_config_path, run_config)

    latest = load_latest_jsonl(responses_path)
    completed_ids = {
        sample_id
        for sample_id, record in latest.items()
        if not record.get("error") and record.get("response") is not None
    }
    expected_ids: list[str] = []
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
                    expected_ids.append(sample.id)
                    if sample.id in completed_ids:
                        progress.update(1)
                        continue
                    future = executor.submit(_call_api, client, sample, sequence)
                    pending[future] = (sample, sequence)
                    if len(pending) >= max_pending:
                        _drain_one(pending, response_file, progress)

                while pending:
                    _drain_one(pending, response_file, progress)
            finally:
                progress.close()

    latest = load_latest_jsonl(responses_path)
    missing = [sample_id for sample_id in expected_ids if sample_id not in latest]
    if missing:
        raise RuntimeError(f"missing API records for {len(missing)} samples")

    records = [latest[sample_id] for sample_id in expected_ids]
    evaluated, summary = adapter.evaluate_with_context(
        records,
        judge_client=judge_client,
        output_dir=output_path,
    )
    summary["num_api_errors"] = sum(bool(item.get("error")) for item in evaluated)
    atomic_write_jsonl(predictions_path, evaluated)
    atomic_write_json(summary_path, summary)
    return summary


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
        "prompt": sample.audit_prompt(),
        "answer": sample.answer,
        "question_type": sample.question_type,
        "choices": sample.choices,
        "index_to_answer": sample.index_to_answer,
        "metadata": sample.metadata,
        "request_options": sample.request_options,
        "response": response,
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 6),
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


def _validate_or_write_config(path: Path, current: dict[str, Any]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != current:
            raise ValueError(
                f"output directory already contains a different run: {path.parent}"
            )
        return
    atomic_write_json(path, current)
