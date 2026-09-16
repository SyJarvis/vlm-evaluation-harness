from __future__ import annotations

import random
from typing import Any, Iterable

from ..data import is_consolidated_local_dataset, load_rows
from ..prompts import MMMU_MULTIPLE_CHOICE, MMMU_SHORT_ANSWER
from ..sample import EvalSample
from ..utils import choice_mapping, numbered_image_content, parse_options
from .base import DatasetAdapter, DatasetConfig
from .common import eval_open_answer, grouped_accuracy, parse_multiple_choice, parse_open_response


MMMU_DOMAINS = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}
MMMU_SUBJECTS = [subject for subjects in MMMU_DOMAINS.values() for subject in subjects]
SUBJECT_TO_DOMAIN = {
    subject: domain for domain, subjects in MMMU_DOMAINS.items() for subject in subjects
}


class MMMUAdapter(DatasetAdapter):
    name = "mmmu"
    hf_path = "MMMU/MMMU"
    default_subset = "all"
    default_split = "validation"

    def iter_rows(self, config: DatasetConfig) -> Iterable[dict[str, Any]]:
        if config.source == "local" and is_consolidated_local_dataset(
            config.dataset_path
        ):
            for row in load_rows(
                source=config.source,
                dataset_path=config.dataset_path,
                subset=None,
                split=config.split,
                cache_dir=config.cache_dir,
                revision=config.revision,
            ):
                yield dict(row)
            return

        subjects = MMMU_SUBJECTS if config.subset == "all" else [config.subset]
        for subject in subjects:
            rows = load_rows(
                source=config.source,
                dataset_path=config.dataset_path,
                subset=subject,
                split=config.split,
                cache_dir=config.cache_dir,
                revision=config.revision,
            )
            for row in rows:
                item = dict(row)
                item["__subject__"] = subject
                yield item

    def build_sample(self, row: dict[str, Any], config: DatasetConfig) -> EvalSample:
        question_type = str(row.get("question_type", "multiple-choice"))
        options = parse_options(row.get("options"))
        choices, index_to_answer = choice_mapping(options)
        question = str(row["question"])
        if question_type == "multiple-choice":
            option_text = "".join(
                f"({label}) {index_to_answer[label]}\n" for label in choices
            ).rstrip()
            prompt = MMMU_MULTIPLE_CHOICE.format(
                question=question, options=option_text
            )
        else:
            prompt = MMMU_SHORT_ANSWER.format(question=question)

        subject = row.get("__subject__") or _subject_from_id(str(row["id"]))
        return EvalSample(
            id=str(row["id"]),
            content=numbered_image_content(prompt, row),
            answer=row.get("answer"),
            question_type=question_type,
            choices=choices,
            index_to_answer=index_to_answer,
            metadata={
                "subject": subject,
                "domain": SUBJECT_TO_DOMAIN.get(subject),
                "subfield": row.get("subfield"),
                "topic_difficulty": row.get("topic_difficulty"),
            },
        )

    def evaluate(self, records):
        rng = random.Random(42)
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction: Any = None
            correct = False
            if not record.get("error"):
                if record["question_type"] == "multiple-choice":
                    prediction = parse_multiple_choice(
                        record.get("response", ""),
                        record["choices"],
                        record["index_to_answer"],
                        rng=rng,
                    )
                    correct = prediction == record.get("answer")
                else:
                    prediction = parse_open_response(record.get("response", ""))
                    correct = eval_open_answer(record.get("answer"), prediction)
            evaluated.append({**record, "prediction": prediction, "correct": correct})

        total = len(evaluated)
        summary = {
            "dataset": self.name,
            "num_samples": total,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total
            if total
            else 0.0,
            "by_subject": grouped_accuracy(evaluated, "subject"),
            "by_domain": grouped_accuracy(evaluated, "domain"),
        }
        return evaluated, summary


def _subject_from_id(sample_id: str) -> str | None:
    for subject in MMMU_SUBJECTS:
        if f"_{subject}_" in sample_id:
            return subject
    return None
