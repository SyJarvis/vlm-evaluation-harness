import json

import pytest
import yaml

from vlm_eval import cli
from vlm_eval.sample import ContentPart, EvalSample
from vlm_eval.tasks.base import DatasetAdapter


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
            {**record, "correct": record.get("response") == record["answer"]}
            for record in records
        ]
        return evaluated, {
            "dataset": self.name,
            "accuracy": sum(item["correct"] for item in evaluated) / len(evaluated),
        }


class FakeChatClient:
    instances = []
    fail_ids: set = set()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChatClient.instances.append(self)

    def complete(self, sample):
        if sample.id in FakeChatClient.fail_ids:
            raise RuntimeError("boom")
        return {"1": "A", "2": "B"}[sample.id]

    def public_config(self):
        return {"model": self.kwargs["model"], "max_tokens": self.kwargs["max_tokens"]}


@pytest.fixture
def fake_api(monkeypatch):
    FakeChatClient.instances = []
    FakeChatClient.fail_ids = set()
    monkeypatch.setattr(cli, "OpenAIChatClient", FakeChatClient)
    monkeypatch.setattr(cli, "get_adapter", lambda name: TinyAdapter())
    return FakeChatClient


def _write_config(tmp_path):
    config = {
        "run": {"name": "suite", "output_root": str(tmp_path / "runs")},
        "model": {"name": "qwen3.5-vl"},
        "tasks": [
            "mmstar",
            {"id": "mmlu_long", "name": "mmlu", "generation": {"max_tokens": 512}},
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _only_run_dir(tmp_path):
    (run_dir,) = (tmp_path / "runs").iterdir()
    return run_dir


def test_run_creates_run_directory(tmp_path, fake_api, capsys):
    cli.main(["run", str(_write_config(tmp_path)), "--limit", "2"])

    run_dir = _only_run_dir(tmp_path)
    assert run_dir.name.endswith("__suite")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert {entry["status"] for entry in manifest["tasks"].values()} == {"completed"}
    assert manifest["tasks"]["mmlu_long"]["context"]["api"]["max_tokens"] == 512
    summary = json.loads((run_dir / "summary.json").read_text())
    assert [row["task_id"] for row in summary["tasks"]] == ["mmstar", "mmlu_long"]
    assert summary["tasks"][0]["metrics"] == {"accuracy": 1.0}
    for task_id in ("mmstar", "mmlu_long"):
        task_dir = run_dir / "tasks" / task_id
        for name in ("samples.jsonl", "responses.jsonl", "predictions.jsonl", "summary.json"):
            assert (task_dir / name).exists()
    assert [client.kwargs["max_tokens"] for client in fake_api.instances] == [128, 512]
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["tasks"][0]["runtime"]["limit"] == 2
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1]) == summary


def test_resume_continues_unfinished_tasks(tmp_path, fake_api):
    cli.main(["run", str(_write_config(tmp_path))])
    run_dir = _only_run_dir(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["tasks"]["mmlu_long"]["status"] = "failed"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    fake_api.instances = []

    cli.main(["resume", str(run_dir)])

    assert [client.kwargs["max_tokens"] for client in fake_api.instances] == [512]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["tasks"]["mmlu_long"]["status"] == "completed"


def test_resume_retries_api_errors_in_completed_tasks(tmp_path, fake_api):
    fake_api.fail_ids = {"2"}
    cli.main(["run", str(_write_config(tmp_path))])
    run_dir = _only_run_dir(tmp_path)
    first = json.loads((run_dir / "tasks" / "mmstar" / "summary.json").read_text())
    fake_api.fail_ids = set()

    cli.main(["resume", str(run_dir)])

    second = json.loads((run_dir / "tasks" / "mmstar" / "summary.json").read_text())
    assert (first["num_errors"], first["metrics"]["accuracy"]) == (1, 0.5)
    assert (second["num_errors"], second["metrics"]["accuracy"]) == (0, 1.0)


def test_resume_rejects_edited_config(tmp_path, fake_api):
    cli.main(["run", str(_write_config(tmp_path))])
    run_dir = _only_run_dir(tmp_path)
    config_path = run_dir / "config.yaml"
    config_path.write_text(config_path.read_text().replace("max_tokens: 512", "max_tokens: 64"))

    with pytest.raises(SystemExit):
        cli.main(["resume", str(run_dir)])


def test_run_rejects_missing_judge_key_before_creating_run(tmp_path, fake_api, monkeypatch):
    monkeypatch.delenv("OPENAI_JUDGE_API_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path / "runs")},
                "model": {"name": "qwen3.5-vl"},
                "tasks": ["charxiv"],
            }
        )
    )

    with pytest.raises(SystemExit):
        cli.main(["run", str(path)])
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("legacy_option", ["--model", "--tasks", "--output-dir"])
def test_flat_options_are_rejected(legacy_option, capsys):
    with pytest.raises(SystemExit):
        cli.main([legacy_option, "x"])
    assert "vlm-eval run" in capsys.readouterr().err
