from __future__ import annotations

from pathlib import Path
from typing import Any


class DatasetLoadError(RuntimeError):
    pass


def is_consolidated_local_dataset(path: str) -> bool:
    local_path = Path(path).expanduser()
    if local_path.is_file():
        return True
    return any(
        (local_path / marker).exists()
        for marker in ("state.json", "dataset_dict.json")
    )


def load_rows(
    *,
    source: str,
    dataset_path: str,
    subset: str | None,
    split: str,
    cache_dir: str | None,
    revision: str | None = None,
):
    if source == "hf":
        from datasets import load_dataset

        return load_dataset(
            dataset_path,
            name=subset,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
    if source == "local":
        return _load_local(dataset_path, subset, split, cache_dir)
    if source == "modelscope":
        return _load_modelscope(dataset_path, subset, split, cache_dir, revision)
    raise DatasetLoadError(f"unsupported dataset source: {source}")


def _load_local(
    dataset_path: str,
    subset: str | None,
    split: str,
    cache_dir: str | None,
):
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise DatasetLoadError(f"local dataset path does not exist: {path}")

    if path.is_dir() and any(
        (path / marker).exists() for marker in ("state.json", "dataset_dict.json")
    ):
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise DatasetLoadError(
                    f"split {split!r} is not present in {path}; available: {list(dataset)}"
                )
            return dataset[split]
        return dataset

    if path.is_file():
        extension = path.suffix.lower().lstrip(".")
        builder = {"jsonl": "json", "tsv": "csv"}.get(extension, extension)
        kwargs: dict[str, Any] = {}
        if extension == "tsv":
            kwargs["delimiter"] = "\t"
        if builder not in {"json", "csv", "parquet", "arrow"}:
            raise DatasetLoadError(f"unsupported local dataset file: {path}")
        return load_dataset(
            builder,
            data_files={split: str(path)},
            split=split,
            cache_dir=cache_dir,
            **kwargs,
        )

    return load_dataset(
        str(path),
        name=subset,
        split=split,
        cache_dir=cache_dir,
    )


def _load_modelscope(
    dataset_path: str,
    subset: str | None,
    split: str,
    cache_dir: str | None,
    revision: str | None,
):
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise DatasetLoadError(
            "ModelScope support requires `pip install -e '.[modelscope]'`"
        ) from exc

    kwargs = {
        "subset_name": subset,
        "split": split,
        "cache_dir": cache_dir,
    }
    if revision is not None:
        kwargs["version"] = revision
    dataset = MsDataset.load(dataset_path, **kwargs)
    if isinstance(dataset, dict):
        if split not in dataset:
            raise DatasetLoadError(
                f"ModelScope returned splits {list(dataset)}, missing {split!r}"
            )
        dataset = dataset[split]
    if hasattr(dataset, "to_hf_dataset"):
        dataset = dataset.to_hf_dataset()
    return dataset
