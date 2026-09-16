from __future__ import annotations

import re
from typing import Any

from ..data import load_rows
from ..prompts import MMSTAR_ANSWER_HINT
from ..sample import ContentPart, EvalSample
from ..utils import choice_mapping
from .base import DatasetAdapter, DatasetConfig
from .common import eval_open_answer, grouped_accuracy, parse_multiple_choice, parse_open_response


class BabyVisionAdapter(DatasetAdapter):
    name = "babyvision"
    hf_path = "UnipatAI/BabyVision"
    default_subset = "default"
    default_split = "train"

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
        question = str(row.get("question", "")).strip()
        ans_type = row.get("ansType", "blank")
        image = row.get("image")
        if image is None:
            raise ValueError(f"BabyVision sample {row.get('taskId')} has no image")

        options = row.get("options") or []
        if ans_type == "choice" and options:
            letters = ["A", "B", "C", "D", "E", "F"][: len(options)]
            # If options are generic descriptions like 'The first one...', just format A. ...
            option_lines = [f"({letter}) {opt}" for letter, opt in zip(letters, options)]
            prompt = f"{question}\n" + "\n".join(option_lines) + f"\n\n{MMSTAR_ANSWER_HINT}"
            
            # choiceAns is 1-indexed or 0-indexed? In HF inspect: choiceAns: 3 with 4 options
            # choiceAns in BabyVision dataset is 1-indexed (1 -> A, 2 -> B, 3 -> C, 4 -> D)
            raw_ans = row.get("choiceAns")
            if isinstance(raw_ans, int):
                if 1 <= raw_ans <= len(letters):
                    answer = letters[raw_ans - 1]
                elif 0 <= raw_ans < len(letters):
                    answer = letters[raw_ans]
                else:
                    answer = str(raw_ans)
            else:
                answer = str(raw_ans or "").strip().upper()

            choices, index_to_answer = choice_mapping([str(opt) for opt in options])
            question_type = "multiple-choice"
        else:
            prompt = f"{question}\nAnswer the question using a single word, number, coordinate, or short phrase directly."
            answer = str(row.get("blankAns", "")).strip()
            choices = []
            index_to_answer = {}
            question_type = "short-answer"

        return EvalSample(
            id=str(row.get("taskId") or row.get("id")),
            content=[ContentPart.image(image), ContentPart.text(prompt)],
            answer=answer,
            question_type=question_type,
            choices=choices,
            index_to_answer=index_to_answer,
            metadata={
                "type": row.get("type"),
                "subtype": row.get("subtype"),
                "ans_type": ans_type,
            },
        )

    def evaluate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        evaluated = []
        for record in sorted(records, key=lambda item: item["sequence"]):
            prediction = None
            correct = False
            if not record.get("error"):
                response_text = record.get("response", "")
                if record.get("question_type") == "multiple-choice":
                    prediction = parse_multiple_choice(
                        response_text,
                        record["choices"],
                        record["index_to_answer"],
                    )
                    correct = prediction == record.get("answer")
                else:
                    predictions = parse_open_response(response_text)
                    gold = record.get("answer", "")
                    # Match normalized or exact coordinates/text
                    correct = _eval_babyvision_blank(gold, response_text, predictions)
                    prediction = predictions[0] if predictions else response_text.strip()
            evaluated.append({**record, "prediction": prediction, "correct": correct})

        total = len(evaluated)
        return evaluated, {
            "dataset": self.name,
            "num_samples": total,
            "accuracy": sum(bool(item["correct"]) for item in evaluated) / total if total else 0.0,
            "by_type": grouped_accuracy(evaluated, "type"),
            "by_subtype": grouped_accuracy(evaluated, "subtype"),
            "by_ans_type": grouped_accuracy(evaluated, "ans_type"),
        }


def _eval_babyvision_blank(gold: str, raw_response: str, parsed_predictions: list[Any]) -> bool:
    gold_clean = str(gold).strip().lower()
    if not gold_clean:
        return False

    # Check coordinate pattern like (x,y)
    coord_pattern = r"\(?\s*(\d+)\s*[,，]\s*(\d+)\s*\)?"
    gold_coord = re.findall(coord_pattern, gold_clean)
    if gold_coord:
        gx, gy = gold_coord[0]
        resp_coords = re.findall(coord_pattern, raw_response.lower())
        if any(rx == gx and ry == gy for rx, ry in resp_coords):
            return True

    # Fallback to standard open answer evaluation
    return eval_open_answer(gold, parsed_predictions)
