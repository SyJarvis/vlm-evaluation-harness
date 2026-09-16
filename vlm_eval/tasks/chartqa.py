from __future__ import annotations

import re
from typing import Any

from ..data import load_rows
from ..prompts import CHARTQA_DIRECT_PROMPT
from ..sample import ContentPart, EvalSample
from .base import DatasetAdapter, DatasetConfig
from .common import calculate_relaxed_accuracy, grouped_accuracy


class ChartQAAdapter(DatasetAdapter):
    name = "chartqa"
    hf_path = "HuggingFaceM4/ChartQA"
    default_subset = "default"
    default_split = "test"

    def iter_rows(self, config: DatasetConfig):
        rows = load_rows(
            source=config.source,
            dataset_path=config.dataset_path,
            subset=config.subset,
            split=config.split,
            cache_dir=config.cache_dir,
            revision=config.revision,
        )
        for row in rows:
            yield dict(row)

    def build_sample(self, row: dict[str, Any], config: DatasetConfig) -> EvalSample:
        question = str(row.get("query") or row.get("question") or "").strip()
        prompt = f"{question}\n{CHARTQA_DIRECT_PROMPT}".strip()
        image = row.get("image")
        if image is None:
            raise ValueError(f"ChartQA sample {row.get('id')} has no image")

        # Answer in ChartQA can be a string or list/float
        raw_label = row.get("label")
        if raw_label is None:
            raw_label = row.get("answer")
        if isinstance(raw_label, list):
            answer = [str(x) for x in raw_label]
        else:
            answer = str(raw_label).strip() if raw_label is not None else ""

        # Determine split/category: human vs augmented
        # Standard ChartQA splits rows into human and augmented in metadata or ID/type
        sample_type = row.get("type") or row.get("source") or "human"
        sample_id = str(row.get("id") or row.get("index") or row.get("imgname") or "")

        return EvalSample(
            id=sample_id,
            content=[ContentPart.image(image), ContentPart.text(prompt)],
            answer=answer,
            question_type="short-answer",
            metadata={
                "type": sample_type,
            },
        )

    def evaluate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction = None
            correct = False
            if not record.get("error"):
                response_text = record.get("response", "")
                prediction = _extract_chartqa_answer(response_text)
                correct = calculate_relaxed_accuracy(prediction, record.get("answer"))
            evaluated.append({**record, "prediction": prediction, "correct": correct})

        total = len(evaluated)
        summary = {
            "dataset": self.name,
            "num_samples": total,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total
            if total
            else 0.0,
            "relaxed_accuracy": sum(bool(item["correct"]) for item in evaluated) / total
            if total
            else 0.0,
            "by_type": grouped_accuracy(evaluated, "type"),
        }
        return evaluated, summary


def _extract_chartqa_answer(response: str) -> str:
    """Extract final answer from direct or reasoning output."""
    if not response:
        return ""
    text = response.strip()
    match = re.findall(r"(?:answer|result)\s*(?:is|:)\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        text = match[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]
    # Strip common leading prefixes if any
    prefix_match = re.match(r"^(?:the answer is|answer:)\s*(.+)$", text, flags=re.IGNORECASE)
    if prefix_match:
        text = prefix_match.group(1).strip()
    return text.strip().rstrip(".")
