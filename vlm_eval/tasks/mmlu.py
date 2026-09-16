from __future__ import annotations

from typing import Any

from ..data import load_rows
from ..prompts import MMSTAR_ANSWER_HINT
from ..sample import ContentPart, EvalSample
from ..utils import choice_mapping
from .base import DatasetAdapter, DatasetConfig
from .common import grouped_accuracy, parse_multiple_choice


class MMLUAdapter(DatasetAdapter):
    name = "mmlu"
    hf_path = "cais/mmlu"
    default_subset = "all"
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
        question = str(row["question"]).strip()
        raw_choices = row.get("choices") or []
        option_letters = ["A", "B", "C", "D"][:len(raw_choices)]
        option_text = "\n".join(
            f"({letter}) {opt}" for letter, opt in zip(option_letters, raw_choices)
        )
        prompt = f"{question}\n{option_text}\n\n{MMSTAR_ANSWER_HINT}"

        raw_ans = row.get("answer")
        if isinstance(raw_ans, int) and 0 <= raw_ans < len(option_letters):
            answer_letter = option_letters[raw_ans]
        else:
            answer_letter = str(raw_ans).strip().upper()

        choices, index_to_answer = choice_mapping([str(c) for c in raw_choices])
        sample_id = str(row.get("id") or f"{row.get('subject', 'sub')}_{hash(question) & 0xFFFFFFFF}")

        return EvalSample(
            id=sample_id,
            content=[ContentPart.text(prompt)],
            answer=answer_letter,
            question_type="multiple-choice",
            choices=choices,
            index_to_answer=index_to_answer,
            metadata={
                "subject": row.get("subject"),
            },
        )

    def evaluate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction = None
            correct = False
            if not record.get("error"):
                prediction = parse_multiple_choice(
                    record.get("response", ""),
                    record["choices"],
                    record["index_to_answer"],
                )
                correct = prediction == record.get("answer")
            evaluated.append({**record, "prediction": prediction, "correct": correct})

        total = len(evaluated)
        summary = {
            "dataset": self.name,
            "num_samples": total,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total if total else 0.0,
            "by_subject": grouped_accuracy(evaluated, "subject"),
        }
        return evaluated, summary
