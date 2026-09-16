from __future__ import annotations

from typing import Any

from ..data import load_rows
from ..prompts import MMSTAR_ANSWER_HINT
from ..sample import ContentPart, EvalSample
from ..utils import choice_mapping
from .base import DatasetAdapter, DatasetConfig
from .common import grouped_accuracy, parse_multiple_choice


class MMStarAdapter(DatasetAdapter):
    name = "mmstar"
    hf_path = "Lin-Chen/MMStar"
    default_subset = "val"
    default_split = "val"

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
        legacy_options = [row.get(label) for label in "ABCD"]
        if all(option is not None for option in legacy_options):
            option_text = "\n".join(
                f"{label}. {option}" for label, option in zip("ABCD", legacy_options)
            )
            question = f"Question: {question}\nOptions:\n{option_text}"
        prompt = f"{question}\n{MMSTAR_ANSWER_HINT}"
        if all(option is not None for option in legacy_options):
            choices, index_to_answer = choice_mapping(
                [str(option) for option in legacy_options]
            )
        else:
            choices, index_to_answer = list("ABCD"), {}
        image = row.get("image")
        if image is None:
            raise ValueError(f"MMStar sample {row.get('index')} has no image")
        return EvalSample(
            id=str(row["index"]),
            content=[ContentPart.image(image), ContentPart.text(prompt)],
            answer=str(row.get("answer", "")).upper(),
            question_type="multiple-choice",
            choices=choices,
            index_to_answer=index_to_answer,
            metadata={
                "category": row.get("category"),
                "l2_category": row.get("l2_category"),
            },
        )

    def evaluate(self, records):
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
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total
            if total
            else 0.0,
            "by_category": grouped_accuracy(evaluated, "category"),
            "by_l2_category": grouped_accuracy(evaluated, "l2_category"),
        }
        return evaluated, summary
