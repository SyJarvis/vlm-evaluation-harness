from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..sample import EvalSample


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    dataset_path: str
    subset: str
    split: str
    cache_dir: str | None = None
    prompt_mode: str = "direct"
    revision: str | None = None


class DatasetAdapter(ABC):
    name: str
    hf_path: str
    default_subset: str
    default_split: str

    def default_path(self, source: str) -> str:
        if source == "hf":
            return self.hf_path
        raise ValueError(f"--dataset-path is required for source {source!r}")

    def protocol_config(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def iter_rows(self, config: DatasetConfig) -> Iterable[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_sample(self, row: dict[str, Any], config: DatasetConfig) -> EvalSample:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def evaluate_with_context(
        self,
        records: list[dict[str, Any]],
        *,
        judge_client: Any | None = None,
        output_dir: Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self.evaluate(records)
