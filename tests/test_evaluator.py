import json

import pytest

from vlm_eval.evaluator import build_task_summary, run_evaluation
from vlm_eval.sample import ContentPart, EvalSample
from vlm_eval.tasks.base import DatasetAdapter, DatasetConfig

CONFIG = DatasetConfig(source="local", dataset_path="unused", subset="default", split="test")


class TinyAdapter(DatasetAdapter):
    name = "tiny"
    hf_path = "unused"
    default_subset = "default"
    default_split = "test"

    def __init__(self, answers=("A", "B")):
        self.answers = answers
        self.iterations = 0

    def iter_rows(self, config):
        self.iterations += 1
        for index, answer in enumerate(self.answers, start=1):
            yield {"id": str(index), "answer": answer}

    def build_sample(self, row, config):
        return EvalSample(
            id=row["id"],
            content=[ContentPart.text(row["id"])],
            answer=row["answer"],
            question_type="multiple-choice",
            metadata={"category": "x"},
        )

    def evaluate(self, records):
        evaluated = [
            {
                **record,
                "prediction": record.get("response"),
                "correct": record.get("response") == record["answer"],
            }
            for record in records
        ]
        return evaluated, {
            "dataset": self.name,
            "num_samples": len(evaluated),
            "accuracy": sum(record["correct"] for record in evaluated) / len(evaluated),
            "by_category": {"x": {"num": len(evaluated), "accuracy": 1.0}},
        }


class RecordingClient:
    def __init__(self, fail_ids=()):
        self.calls = []
        self.fail_ids = set(fail_ids)

    def complete(self, sample):
        self.calls.append(sample.id)
        if sample.id in self.fail_ids:
            raise RuntimeError("boom")
        return {"1": "A", "2": "B"}[sample.id]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(tmp_path, adapter, client, **kwargs):
    return run_evaluation(
        adapter=adapter,
        dataset_config=CONFIG,
        client=client,
        output_dir=tmp_path,
        concurrency=2,
        **kwargs,
    )


def test_evaluator_splits_samples_responses_and_predictions(tmp_path):
    summary = _run(tmp_path, TinyAdapter(), RecordingClient(), task_id="tiny_run")

    samples = _read_jsonl(tmp_path / "samples.jsonl")
    responses = _read_jsonl(tmp_path / "responses.jsonl")
    predictions = _read_jsonl(tmp_path / "predictions.jsonl")
    assert [sample["id"] for sample in samples] == ["1", "2"]
    assert set(samples[0]) == {
        "id",
        "sequence",
        "prompt",
        "answer",
        "question_type",
        "choices",
        "index_to_answer",
        "metadata",
    }
    assert set(responses[0]) == {
        "id",
        "sequence",
        "response",
        "error",
        "latency_seconds",
        "request_options",
    }
    assert predictions[0]["answer"] == "A"
    assert predictions[0]["response"] == "A"
    assert predictions[0]["correct"] is True
    assert summary == json.loads((tmp_path / "summary.json").read_text())
    assert summary["task_id"] == "tiny_run"
    assert summary["metrics"] == {"accuracy": 1.0}
    assert summary["groups"] == {"category": {"x": {"num": 2, "accuracy": 1.0}}}
    assert summary["num_errors"] == 0


def test_completed_run_resumes_without_reading_dataset(tmp_path):
    client = RecordingClient()
    _run(tmp_path, TinyAdapter(), client)
    adapter = TinyAdapter()

    summary = _run(tmp_path, adapter, client)

    assert adapter.iterations == 0
    assert sorted(client.calls) == ["1", "2"]
    assert summary["metrics"]["accuracy"] == 1.0


def test_failed_samples_are_retried_on_resume(tmp_path):
    first = _run(tmp_path, TinyAdapter(), RecordingClient(fail_ids={"2"}))
    retry_client = RecordingClient()

    second = _run(tmp_path, TinyAdapter(), retry_client)

    assert first["num_errors"] == 1
    assert first["metrics"]["accuracy"] == 0.5
    assert retry_client.calls == ["2"]
    assert second["num_errors"] == 0
    assert second["metrics"]["accuracy"] == 1.0
    assert len(_read_jsonl(tmp_path / "responses.jsonl")) == 3


def test_resume_rejects_changed_dataset(tmp_path):
    _run(tmp_path, TinyAdapter(), RecordingClient(fail_ids={"2"}))

    with pytest.raises(ValueError, match="dataset no longer matches"):
        _run(tmp_path, TinyAdapter(answers=("C", "B")), RecordingClient())


def test_evaluator_passes_judge_context(tmp_path):
    class ContextAdapter(TinyAdapter):
        def evaluate_with_context(self, records, *, judge_client=None, output_dir=None):
            self.context = (judge_client, output_dir)
            return self.evaluate(records)

    adapter = ContextAdapter()
    judge = object()

    _run(tmp_path, adapter, RecordingClient(), judge_client=judge)

    assert adapter.context == (judge, tmp_path)


def test_task_summary_lifts_charxiv_overall_scores():
    raw = {
        "dataset": "charxiv",
        "num_samples": 5,
        "num_charts": 1,
        "scored": True,
        "descriptive": {"Overall Score": 80.0, "N_valid": 4},
        "reasoning": {"Overall Score": 100.0, "N_valid": 1},
    }

    summary = build_task_summary("charxiv", "charxiv", raw, [{"error": None}] * 5)

    assert summary["metrics"] == {"descriptive_overall": 80.0, "reasoning_overall": 100.0}
    assert summary["descriptive"]["N_valid"] == 4
    assert summary["num_charts"] == 1
    assert summary["scored"] is True
