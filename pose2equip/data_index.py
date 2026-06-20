"""Utilities for loading precomputed Unity fold index files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pose2equip.map_config import UnityDataConfig


def remap_unity_dataset_paths(value: Any, unity_root: Path) -> Any:
    """Remap serialized Unity dataset paths to the configured local root."""
    dataset_dirname = unity_root.name
    if isinstance(value, str):
        marker = f"/{dataset_dirname}/"
        if marker in value:
            suffix = value.split(marker, 1)[1]
            return str(unity_root / suffix)
        if value.endswith(f"/{dataset_dirname}"):
            return str(unity_root)
        return value
    if isinstance(value, dict):
        return {k: remap_unity_dataset_paths(v, unity_root) for k, v in value.items()}
    if isinstance(value, list):
        return [remap_unity_dataset_paths(v, unity_root) for v in value]
    return value


def fold_file_for(index_mapping_path: str | Path, fold: int) -> Path:
    """Return the expected fold JSON path for a fold number."""
    return Path(index_mapping_path) / f"fold_{int(fold):02d}.json"


def detect_available_folds(config: Any, index_mapping_path: str | Path | None = None) -> List[int]:
    """Detect available fold numbers from fold_XX.json files."""
    index_path = Path(index_mapping_path or str(config.data.index_mapping_path))
    available_folds: List[int] = []
    for fold_file in sorted(index_path.glob("fold_*.json")):
        suffix = fold_file.stem.replace("fold_", "")
        if suffix.isdigit():
            available_folds.append(int(suffix))
    return sorted(available_folds)


def load_fold_dataset_idx_from_fold_json(
    config: Any,
    fold: int,
    index_mapping_path: str | Path | None = None,
    unity_root: str | Path | None = None,
) -> Dict[str, List[UnityDataConfig]]:
    """Load one precomputed fold JSON as UnityDataConfig entries."""
    index_path = Path(index_mapping_path or str(config.data.index_mapping_path))
    root = Path(unity_root or str(config.data.unity.root_path))
    fold_file = fold_file_for(index_path, fold)

    with fold_file.open("r", encoding="utf-8") as f:
        fold_data = json.load(f)
    fold_data.pop("_metadata", None)

    dataset_idx: Dict[str, List[UnityDataConfig]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for item in fold_data.get(split, []):
            remapped = remap_unity_dataset_paths(item, root)
            dataset_idx[split].append(UnityDataConfig.from_dict(remapped))

    return dataset_idx
