from .base import DatasetAdapter, DatasetConfig
from .babyvision import BabyVisionAdapter
from .chartqa import ChartQAAdapter
from .charxiv import CharXivAdapter
from .docvqa import DocVQAAdapter
from .mmlu import MMLUAdapter
from .mmmu import MMMUAdapter
from .mmmu_pro import MMMUProAdapter
from .mmstar import MMStarAdapter


ADAPTERS: dict[str, DatasetAdapter] = {
    "babyvision": BabyVisionAdapter(),
    "chartqa": ChartQAAdapter(),
    "docvqa": DocVQAAdapter(),
    "mmlu": MMLUAdapter(),
    "mmmu": MMMUAdapter(),
    "mmmu_pro": MMMUProAdapter(),
    "mmstar": MMStarAdapter(),
    "charxiv": CharXivAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {name}") from exc


__all__ = ["ADAPTERS", "DatasetConfig", "get_adapter"]
