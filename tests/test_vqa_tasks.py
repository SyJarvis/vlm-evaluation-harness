import pytest
from vlm_eval.tasks.chartqa import ChartQAAdapter, _extract_chartqa_answer
from vlm_eval.tasks.common import calculate_anls, calculate_relaxed_accuracy
from vlm_eval.tasks.docvqa import DocVQAAdapter, _extract_docvqa_answer
from vlm_eval.tasks.base import DatasetConfig


def test_calculate_relaxed_accuracy():
    # Exact text
    assert calculate_relaxed_accuracy("apple", "apple")
    assert calculate_relaxed_accuracy("Apple", "apple")
    assert calculate_relaxed_accuracy("apple.", "apple")

    # Numbers within 5%
    assert calculate_relaxed_accuracy("100", "100")
    assert calculate_relaxed_accuracy("104", "100")  # 4% diff <= 5%
    assert calculate_relaxed_accuracy("96", "100")   # 4% diff <= 5%
    assert calculate_relaxed_accuracy("105", "100")  # 5% diff <= 5%
    assert not calculate_relaxed_accuracy("106", "100") # 6% diff > 5%

    # Units / percentage
    assert calculate_relaxed_accuracy("50%", "50")
    assert calculate_relaxed_accuracy("$1000", "1,000")

    # List of golds
    assert calculate_relaxed_accuracy("10", ["5", "10"])
    assert not calculate_relaxed_accuracy("20", ["5", "10"])


def test_calculate_anls():
    # Exact match
    assert calculate_anls("apple", ["apple"]) == 1.0
    assert calculate_anls("Apple", ["apple"]) == 1.0

    # Slight difference (edit distance = 1, max len = 5 -> dist / 5 = 0.2 < 0.5 -> 0.8)
    assert calculate_anls("apply", ["apple"]) == pytest.approx(0.8)

    # Big difference (dist > 0.5 -> 0.0)
    assert calculate_anls("banana", ["apple"]) == 0.0

    # Multiple gold answers
    assert calculate_anls("apply", ["orange", "apple"]) == pytest.approx(0.8)


def test_chartqa_adapter_build_sample():
    adapter = ChartQAAdapter()
    config = DatasetConfig(source="hf", dataset_path="HuggingFaceM4/ChartQA", subset="default", split="test")
    row = {
        "id": "test_0",
        "query": "What is the highest value in 2020?",
        "label": ["42.5"],
        "image": "dummy_image",
        "type": "human",
    }
    sample = adapter.build_sample(row, config)
    assert sample.id == "test_0"
    assert sample.question_type == "short-answer"
    assert "42.5" in sample.answer
    assert sample.metadata["type"] == "human"


def test_chartqa_evaluation():
    adapter = ChartQAAdapter()
    records = [
        {"sequence": 0, "response": "The answer is 100", "answer": "100", "metadata": {"type": "human"}},
        {"sequence": 1, "response": "Answer: 50", "answer": "100", "metadata": {"type": "augmented"}},
    ]
    evaluated, summary = adapter.evaluate(records)
    assert len(evaluated) == 2
    assert evaluated[0]["correct"] is True
    assert evaluated[1]["correct"] is False
    assert summary["num_samples"] == 2
    assert summary["relaxed_accuracy"] == 0.5
    assert summary["by_type"]["human"]["accuracy"] == 1.0
    assert summary["by_type"]["augmented"]["accuracy"] == 0.0


def test_docvqa_adapter_build_sample():
    adapter = DocVQAAdapter()
    config = DatasetConfig(source="hf", dataset_path="lmms-lab/DocVQA", subset="DocVQA", split="validation")
    row = {
        "questionId": 1234,
        "question": "What is the date?",
        "answers": ["11/12/2020", "Nov 12, 2020"],
        "image": "dummy_image",
        "docId": "doc_99",
    }
    sample = adapter.build_sample(row, config)
    assert sample.id == "1234"
    assert sample.question_type == "short-answer"
    assert len(sample.answer) == 2
    assert sample.metadata["doc_id"] == "doc_99"


def test_docvqa_evaluation():
    adapter = DocVQAAdapter()
    records = [
        {"sequence": 0, "response": "11/12/2020", "answer": ["11/12/2020", "Nov 12, 2020"]},
        {"sequence": 1, "response": "wrong date", "answer": ["11/12/2020"]},
    ]
    evaluated, summary = adapter.evaluate(records)
    assert len(evaluated) == 2
    assert evaluated[0]["anls"] == 1.0
    assert evaluated[0]["correct"] is True
    assert evaluated[1]["anls"] == 0.0
    assert evaluated[1]["correct"] is False
    assert summary["anls"] == 0.5
