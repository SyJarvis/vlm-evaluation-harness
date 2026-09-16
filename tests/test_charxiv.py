import json

from PIL import Image

from vlm_eval.tasks.base import DatasetConfig
from vlm_eval.tasks.charxiv import CharXivAdapter


def config(dataset_path="unused", split="validation"):
    return DatasetConfig(
        source="local",
        dataset_path=str(dataset_path),
        subset="default",
        split=split,
    )


def hub_row(image=None):
    return {
        "image": image or Image.new("RGB", (2, 2), "white"),
        "figure_path": "images/7.jpg",
        "category": "cs",
        "year": "20",
        "num_subplots": 2,
        "subplot_row": 1,
        "subplot_col": 2,
        "subplot_loc": None,
        "descriptive_q1": 1,
        "descriptive_q2": 13,
        "descriptive_q3": 18,
        "descriptive_q4": 14,
        "descriptive_a1": "Title",
        "descriptive_a2": "A, B",
        "descriptive_a3": "1 by 2",
        "descriptive_a4": "Not Applicable",
        "reasoning_q": "What is the difference?",
        "reasoning_q_source": 3,
        "reasoning_a": "2.50",
        "reasoning_a_type": 4,
    }


def test_charxiv_expands_chart_and_uses_official_generation_protocol():
    adapter = CharXivAdapter()
    rows = list(adapter._expand_hub_row(hub_row()))

    sample = adapter.build_sample(rows[0], config())
    reasoning = adapter.build_sample(rows[-1], config())

    assert [row["id"] for row in rows] == [
        "descriptive:7_0",
        "descriptive:7_1",
        "descriptive:7_2",
        "descriptive:7_3",
        "reasoning:7",
    ]
    assert [part.type for part in sample.content] == ["text", "image"]
    assert sample.content[0].value.startswith(
        "For the subplot at row 1 and column 2, what is its title?"
    )
    assert sample.request_options == {
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    assert "2 decimal places" in reasoning.content[0].value


def test_charxiv_reads_official_local_repository_layout(tmp_path):
    data_dir = tmp_path / "data"
    image_dir = tmp_path / "images"
    data_dir.mkdir()
    image_dir.mkdir()
    Image.new("RGB", (2, 2), "white").save(image_dir / "7.jpg")
    (data_dir / "descriptive_val.json").write_text(
        json.dumps(
            {
                "7": {
                    "figure_id": 7,
                    "num_subplots": 1,
                    "subplot_loc": [0, 0],
                    "qids": [1, 2, 3, 4],
                    "answers": ["T", "X", "Y", "0"],
                }
            }
        )
    )
    (data_dir / "reasoning_val.json").write_text(
        json.dumps(
            {
                "7": {
                    "figure_id": 7,
                    "query": "Which line?",
                    "answer": "A",
                    "inst_category": 1,
                    "qa_source": 2,
                }
            }
        )
    )
    (data_dir / "image_metadata_val.json").write_text(
        json.dumps({"7": {"category": "math", "year": "23"}})
    )

    rows = list(CharXivAdapter().iter_rows(config(tmp_path)))

    assert len(rows) == 5
    assert rows[0]["image"] == image_dir / "7.jpg"
    assert rows[-1]["raw_question"] == "Which line?"


class RecordingJudge:
    model = "gpt-4o-2024-05-13"

    def __init__(self):
        self.calls = []

    def complete_json(self, prompt, *, max_tokens):
        self.calls.append((prompt, max_tokens))
        if "Overarching Question" in prompt:
            count = prompt.count("Ground Truth ")
            result = {}
            for index in range(1, count + 1):
                result[f"extract_answer_T{index}"] = "correct"
                result[f"score_T{index}"] = 1
            return result
        return {"extracted_answer": "correct", "score": 1}


def record(sample_id, sequence, mode, **metadata):
    return {
        "id": sample_id,
        "sequence": sequence,
        "answer": "correct",
        "response": "correct",
        "error": None,
        "metadata": {
            "mode": mode,
            "figure_id": metadata.pop("figure_id", "7"),
            "num_subplots": 1,
            "category": "cs",
            "year": "20",
            **metadata,
        },
    }


def test_charxiv_judge_batches_scores_and_resumes(tmp_path):
    adapter = CharXivAdapter()
    judge = RecordingJudge()
    records = [
        record("descriptive:7_0", 0, "descriptive", qid=1),
        record("descriptive:8_0", 1, "descriptive", figure_id="8", qid=1),
        record(
            "reasoning:7",
            2,
            "reasoning",
            inst_category=1,
            qa_source=3,
            raw_question="Which line?",
        ),
    ]

    first_records, first_summary = adapter.evaluate_with_context(
        records, judge_client=judge, output_dir=tmp_path
    )
    second_records, second_summary = adapter.evaluate_with_context(
        records, judge_client=judge, output_dir=tmp_path
    )

    assert len(judge.calls) == 2
    assert len((tmp_path / "judge_calls.jsonl").read_text().splitlines()) == 2
    assert [item["score"] for item in first_records] == [1, 1, 1]
    assert first_summary["descriptive"]["Overall Score"] == 100.0
    assert first_summary["reasoning"]["Overall Score"] == 100.0
    assert second_records == first_records
    assert second_summary == first_summary


def test_charxiv_public_test_is_generation_only_without_judge(tmp_path):
    adapter = CharXivAdapter()
    records = [
        {
            **record(
                "reasoning:7",
                0,
                "reasoning",
                inst_category=4,
                qa_source=1,
                raw_question="What value?",
            ),
            "answer": None,
            "metadata": {
                "mode": "reasoning",
                "figure_id": "7",
                "num_subplots": 1,
                "category": "cs",
                "year": "20",
                "inst_category": 4,
                "qa_source": 1,
                "raw_question": "What value?",
                "precision_instruction_omitted": True,
            },
        }
    ]

    evaluated, summary = adapter.evaluate_with_context(records, output_dir=tmp_path)

    assert evaluated[0]["score"] is None
    assert summary["scored"] is False
    assert summary["num_precision_instructions_omitted"] == 1
