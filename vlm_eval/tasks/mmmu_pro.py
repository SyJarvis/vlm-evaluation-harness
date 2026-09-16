from __future__ import annotations

import random
from typing import Any

from ..data import load_rows
from ..prompts import (
    MMMU_PRO_COT_STANDARD,
    MMMU_PRO_COT_VISION,
    MMMU_PRO_DIRECT_STANDARD,
    MMMU_PRO_DIRECT_VISION,
)
from ..sample import ContentPart, EvalSample
from ..utils import choice_mapping, numbered_image_content, parse_options
from .base import DatasetAdapter, DatasetConfig
from .common import grouped_accuracy, parse_multiple_choice


class MMMUProAdapter(DatasetAdapter):
    name = "mmmu_pro"
    hf_path = "MMMU/MMMU_Pro"
    default_subset = "standard (10 options)"
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
        options = parse_options(row.get("options"))
        choices, index_to_answer = choice_mapping(options)
        if config.subset == "vision":
            prompt = (
                MMMU_PRO_COT_VISION
                if config.prompt_mode == "cot"
                else MMMU_PRO_DIRECT_VISION
            )
            image = row.get("image")
            if image is None:
                raise ValueError(f"MMMU-Pro vision sample {row.get('id')} has no image")
            content = [ContentPart.text(prompt), ContentPart.image(image)]
        else:
            suffix = (
                MMMU_PRO_COT_STANDARD
                if config.prompt_mode == "cot"
                else MMMU_PRO_DIRECT_STANDARD
            )
            option_text = "\n".join(
                f"{label}. {index_to_answer[label]}" for label in choices
            )
            prompt = f"{row['question']}\n{option_text}\n{suffix}"
            content = numbered_image_content(prompt, row, fallback_position="after")

        return EvalSample(
            id=str(row["id"]),
            content=content,
            answer=row.get("answer"),
            question_type="multiple-choice",
            choices=choices,
            index_to_answer=index_to_answer,
            metadata={"subject": row.get("subject"), "setting": config.subset},
        )

    def evaluate(self, records):
        rng = random.Random(42)
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction = None
            correct = False
            if not record.get("error"):
                prediction = parse_multiple_choice(
                    record.get("response", ""),
                    record["choices"],
                    record["index_to_answer"],
                    rng=rng,
                )
                correct = prediction == record.get("answer")
            evaluated.append({**record, "prediction": prediction, "correct": correct})

        total = len(evaluated)
        summary = {
            "dataset": self.name,
            "setting": evaluated[0]["metadata"].get("setting") if evaluated else None,
            "num_samples": total,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total
            if total
            else 0.0,
            "by_subject": grouped_accuracy(evaluated, "subject"),
        }
        return evaluated, summary
