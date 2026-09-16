import json

from vlm_eval.evaluator import run_evaluation
from vlm_eval.sample import ContentPart, EvalSample
from vlm_eval.tasks.base import DatasetAdapter, DatasetConfig


class TinyAdapter(DatasetAdapter):
    name = "tiny"
    hf_path = "unused"
    default_subset = "default"
    default_split = "test"

    def iter_rows(self, config):
        yield {"id": "1", "answer": "A"}
        yield {"id": "2", "answer": "B"}

    def build_sample(self, row, config):
        return EvalSample(
            id=row["id"],
            content=[ContentPart.text(row["id"])],
            answer=row["answer"],
            question_type="multiple-choice",
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
            "accuracy": sum(record["correct"] for record in evaluated)
            / len(evaluated),
        }


class RecordingClient:
    def __init__(self):
        self.calls = []

    def complete(self, sample):
        self.calls.append(sample.id)
        return {"1": "A", "2": "B"}[sample.id]

    def public_config(self):
        return {"model": "fake"}


class RecordingJudge:
    def public_config(self):
        return {"model": "judge"}


class ContextAdapter(TinyAdapter):
    def __init__(self):
        self.context = None

    def protocol_config(self):
        return {"version": "test"}

    def evaluate_with_context(
        self, records, *, judge_client=None, output_dir=None
    ):
        self.context = (judge_client, output_dir)
        return self.evaluate(records)


def test_evaluator_writes_results_and_resumes_successful_samples(tmp_path):
    client = RecordingClient()
    config = DatasetConfig(
        source="local",
        dataset_path="unused",
        subset="default",
        split="test",
    )

    first = run_evaluation(
        adapter=TinyAdapter(),
        dataset_config=config,
        client=client,
        output_dir=tmp_path,
        concurrency=2,
    )
    second = run_evaluation(
        adapter=TinyAdapter(),
        dataset_config=config,
        client=client,
        output_dir=tmp_path,
        concurrency=2,
    )

    assert first["accuracy"] == 1.0
    assert second["accuracy"] == 1.0
    assert sorted(client.calls) == ["1", "2"]
    assert (tmp_path / "responses.jsonl").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["num_api_errors"] == 0


def test_evaluator_passes_judge_context_and_audits_protocol(tmp_path):
    adapter = ContextAdapter()
    judge = RecordingJudge()
    config = DatasetConfig(
        source="local",
        dataset_path="unused",
        subset="default",
        split="test",
    )

    run_evaluation(
        adapter=adapter,
        dataset_config=config,
        client=RecordingClient(),
        judge_client=judge,
        output_dir=tmp_path,
    )

    run_config = json.loads((tmp_path / "run_config.json").read_text())
    assert adapter.context == (judge, tmp_path)
    assert run_config["judge_api"] == {"model": "judge"}
    assert run_config["task_protocol"] == {"version": "test"}
