from .base import DatasetAdapter, DatasetConfig
from .charxiv import CharXivAdapter
from .mmmu import MMMUAdapter
from .mmmu_pro import MMMUProAdapter
from .mmstar import MMStarAdapter


ADAPTERS: dict[str, DatasetAdapter] = {
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
