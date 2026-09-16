from .base import DatasetAdapter, DatasetConfig
from .chartqa import ChartQAAdapter
from .charxiv import CharXivAdapter
from .docvqa import DocVQAAdapter
from .mmlu import MMLUAdapter
from .mmmu import MMMUAdapter
from .mmmu_pro import MMMUProAdapter
from .mmstar import MMStarAdapter


ADAPTERS: dict[str, DatasetAdapter] = {
    "chartqa": ChartQAAdapter(),
    "docvqa": DocVQAAdapter(),
    "mmlu": MMLUAdapter(),
    "mmmu": MMMUAdapter(),
    "mmmu_pro": MMMUProAdapter(),
    "mmstar": MMStarAdapter(),
    "charxiv": CharXivAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in ADAPTERS:
        raise KeyError(
            f"unknown task '{name}'. available: {', '.join(sorted(ADAPTERS))}"
        )
    return ADAPTERS[name]


__all__ = ["ADAPTERS", "DatasetConfig", "get_adapter"]
