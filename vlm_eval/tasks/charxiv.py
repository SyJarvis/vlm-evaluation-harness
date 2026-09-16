from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..data import load_rows
from ..sample import ContentPart, EvalSample
from ..task_config import load_task_config
from .base import DatasetAdapter, DatasetConfig


class CharXivAdapter(DatasetAdapter):
    name = "charxiv"

    def __init__(self) -> None:
        self.spec = load_task_config(self.name)
        dataset = self.spec["dataset"]
        self.hf_path = dataset["hf_path"]
        self.default_subset = dataset["default_subset"]
        self.default_split = dataset["default_split"]

    def iter_rows(self, config: DatasetConfig) -> Iterable[dict[str, Any]]:
        if config.source == "local" and self._is_official_local(config):
            yield from self._iter_official_local(config)
            return

        rows = load_rows(
            source=config.source,
            dataset_path=config.dataset_path,
            subset=config.subset,
            split=config.split,
            cache_dir=config.cache_dir,
            revision=config.revision,
        )
        for row in rows:
            yield from self._expand_hub_row(dict(row))

    def protocol_config(self) -> dict[str, Any]:
        serialized = json.dumps(
            self.spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "version": self.spec["protocol_version"],
            "upstream": self.spec["upstream"],
            "config_sha256": hashlib.sha256(serialized).hexdigest(),
        }

    def build_sample(self, row: dict[str, Any], config: DatasetConfig) -> EvalSample:
        if row["mode"] == "descriptive":
            question = self._descriptive_question(row["qid"], row["subplot_loc"])
        else:
            question = self._reasoning_question(
                row["raw_question"], row["inst_category"], row.get("answer")
            )

        image = row.get("image")
        if image is None:
            raise ValueError(f"CharXiv sample {row['id']} has no image")
        if isinstance(image, (str, Path)) and not Path(image).exists():
            raise ValueError(f"CharXiv image does not exist: {image}")

        metadata = {
            key: row.get(key)
            for key in (
                "mode",
                "figure_id",
                "subq_idx",
                "qid",
                "num_subplots",
                "subplot_loc",
                "category",
                "year",
                "inst_category",
                "qa_source",
                "raw_question",
            )
            if row.get(key) is not None
        }
        if row["mode"] == "reasoning" and row.get("answer") is None:
            metadata["precision_instruction_omitted"] = row["inst_category"] == 4

        return EvalSample(
            id=row["id"],
            content=[ContentPart.text(question), ContentPart.image(image)],
            answer=row.get("answer"),
            question_type="open-ended",
            metadata=metadata,
            request_options=dict(self.spec["generation"]),
        )

    def evaluate(self, records):
        if all(record.get("answer") is None for record in records):
            return self._unscored(records)
        raise ValueError("CharXiv validation scoring requires a judge client")

    def evaluate_with_context(
        self,
        records,
        *,
        judge_client=None,
        output_dir: Path | None = None,
    ):
        if all(record.get("answer") is None for record in records):
            return self._unscored(records)
        if any(record.get("answer") is None for record in records):
            raise ValueError("CharXiv records cannot mix public test and validation data")
        if judge_client is None:
            raise ValueError("CharXiv validation scoring requires --judge-api-key")
        if output_dir is None:
            raise ValueError("CharXiv scoring requires an output directory")

        from .charxiv_score import score_charxiv

        return score_charxiv(
            records,
            spec=self.spec,
            judge_client=judge_client,
            output_dir=output_dir,
        )

    def _is_official_local(self, config: DatasetConfig) -> bool:
        root = Path(config.dataset_path).expanduser()
        split = self._official_split(config.split)
        relative = self.spec["dataset"]["official_local_files"]["descriptive"]
        return root.is_dir() and (root / relative.format(split=split)).exists()

    def _iter_official_local(self, config: DatasetConfig):
        root = Path(config.dataset_path).expanduser().resolve()
        split = self._official_split(config.split)
        files = self.spec["dataset"]["official_local_files"]
        descriptive = self._read_json(root / files["descriptive"].format(split=split))
        reasoning = self._read_json(root / files["reasoning"].format(split=split))
        image_metadata = self._read_json(
            root / files["image_metadata"].format(split=split)
        )
        image_dir = root / files["images"]

        for item in descriptive.values():
            figure_id = str(item["figure_id"])
            common = {
                "figure_id": figure_id,
                "image": image_dir / f"{figure_id}.jpg",
                "num_subplots": item["num_subplots"],
                "subplot_loc": item["subplot_loc"],
                "category": image_metadata[figure_id]["category"],
                "year": image_metadata[figure_id]["year"],
            }
            answers = item.get("answers")
            for index, qid in enumerate(item["qids"]):
                yield {
                    **common,
                    "id": f"descriptive:{figure_id}_{index}",
                    "mode": "descriptive",
                    "subq_idx": index,
                    "qid": int(qid),
                    "answer": None if answers is None else answers[index],
                }

            reason = reasoning[figure_id]
            yield {
                **common,
                "id": f"reasoning:{figure_id}",
                "mode": "reasoning",
                "raw_question": reason["query"],
                "answer": reason.get("answer"),
                "inst_category": int(reason["inst_category"]),
                "qa_source": int(reason["qa_source"]),
            }

    def _expand_hub_row(self, row: dict[str, Any]):
        figure_id = Path(str(row["figure_path"])).stem
        subplot_loc: list[int] | str
        if row.get("subplot_loc"):
            subplot_loc = str(row["subplot_loc"])
        else:
            subplot_loc = [int(row["subplot_row"]), int(row["subplot_col"])]
        common = {
            "figure_id": figure_id,
            "image": row["image"],
            "num_subplots": int(row["num_subplots"]),
            "subplot_loc": subplot_loc,
            "category": row["category"],
            "year": row["year"],
        }
        for index in range(4):
            position = index + 1
            yield {
                **common,
                "id": f"descriptive:{figure_id}_{index}",
                "mode": "descriptive",
                "subq_idx": index,
                "qid": int(row[f"descriptive_q{position}"]),
                "answer": row.get(f"descriptive_a{position}"),
            }
        yield {
            **common,
            "id": f"reasoning:{figure_id}",
            "mode": "reasoning",
            "raw_question": row["reasoning_q"],
            "answer": row.get("reasoning_a"),
            "inst_category": int(row["reasoning_a_type"]),
            "qa_source": int(row["reasoning_q_source"]),
        }

    def _descriptive_question(self, qid: int, subplot_loc: list[int] | str) -> str:
        template = self.spec["descriptive"]["response_instructions"][str(qid)]
        if qid in (18, 19):
            return template
        if isinstance(subplot_loc, list):
            if subplot_loc[0] == 0:
                prefix = "For the current plot, "
            else:
                prefix = (
                    f"For the subplot at row {subplot_loc[0]} "
                    f"and column {subplot_loc[1]}, "
                )
        elif isinstance(subplot_loc, str):
            prefix = f"For {subplot_loc}, "
        else:
            raise ValueError(f"invalid CharXiv subplot location: {subplot_loc!r}")
        return template.format(prefix)

    def _reasoning_question(
        self, question: str, inst_category: int, answer: Any
    ) -> str:
        template = self.spec["reasoning"]["response_instructions"][
            str(inst_category)
        ]
        if inst_category != 4:
            return template.format(question)
        instruction = "" if answer is None else self._number_instruction(str(answer))
        return template.format(question, instruction)

    @staticmethod
    def _number_instruction(answer: str) -> str:
        parts = answer.split(".")
        if len(parts) == 1:
            return "* Your final answer must be an exact integer."
        return f"* Your final answer must be a number with {len(parts[1])} decimal places."

    def _official_split(self, split: str) -> str:
        return self.spec["dataset"]["official_split_aliases"].get(split, split)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _unscored(self, records):
        evaluated = [
            {**record, "prediction": None, "score": None, "correct": None}
            for record in sorted(records, key=lambda item: item["sequence"])
        ]
        modes = ("descriptive", "reasoning")
        summary = {
            "dataset": self.name,
            "num_samples": len(evaluated),
            "num_charts": len(
                {item["metadata"]["figure_id"] for item in evaluated}
            ),
            "scored": False,
            "reason": "CharXiv public test answers are hidden",
            "protocol_comparable": False,
        }
        for mode in modes:
            subset = [item for item in evaluated if item["metadata"]["mode"] == mode]
            summary[mode] = {"num_samples": len(subset)}
        summary["num_precision_instructions_omitted"] = sum(
            bool(item["metadata"].get("precision_instruction_omitted"))
            for item in evaluated
        )
        return evaluated, summary
