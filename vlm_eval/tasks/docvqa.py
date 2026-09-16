from __future__ import annotations

import re
from typing import Any

from ..data import load_rows
from ..prompts import DOCVQA_DIRECT_PROMPT
from ..sample import ContentPart, EvalSample
from .base import DatasetAdapter, DatasetConfig
from .common import calculate_anls


class DocVQAAdapter(DatasetAdapter):
    name = "docvqa"
    hf_path = "lmms-lab/DocVQA"
    default_subset = "DocVQA"
    default_split = "validation"

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
        question = str(row.get("question") or row.get("query") or "").strip()
        prompt = f"{question}\n{DOCVQA_DIRECT_PROMPT}".strip()
        image = row.get("image")
        if image is None:
            raise ValueError(f"DocVQA sample {row.get('questionId')} has no image")

        raw_answers = row.get("answers")
        if raw_answers is None:
            raw_answers = row.get("answer")
        if isinstance(raw_answers, list):
            answers = [str(x) for x in raw_answers]
        elif raw_answers is not None:
            answers = [str(raw_answers)]
        else:
            answers = []

        sample_id = str(row.get("questionId") or row.get("id") or row.get("index") or "")

        return EvalSample(
            id=sample_id,
            content=[ContentPart.image(image), ContentPart.text(prompt)],
            answer=answers,
            question_type="short-answer",
            metadata={
                "doc_id": row.get("docId"),
            },
        )

    def evaluate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction = None
            anls = 0.0
            if not record.get("error"):
                response_text = record.get("response", "")
                prediction = _extract_docvqa_answer(response_text)
                answers = record.get("answer")
                if isinstance(answers, str):
                    answers = [answers]
                elif not isinstance(answers, list):
                    answers = []
                anls = calculate_anls(prediction, answers)
            evaluated.append({**record, "prediction": prediction, "anls": anls, "correct": anls >= 0.5})

        total = len(evaluated)
        summary = {
            "dataset": self.name,
            "num_samples": total,
            "anls": sum(float(item["anls"]) for item in evaluated) / total if total else 0.0,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total if total else 0.0,
        }
        return evaluated, summary


def _extract_docvqa_answer(response: str) -> str:
    """Extract answer for DocVQA."""
    if not response:
        return ""
    text = response.strip()
    match = re.findall(r"(?:answer|result)\s*(?:is|:)\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        text = match[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1]
    prefix_match = re.match(r"^(?:the answer is|answer:)\s*(.+)$", text, flags=re.IGNORECASE)
    if prefix_match:
        text = prefix_match.group(1).strip()
    return text.strip().rstrip(".")
