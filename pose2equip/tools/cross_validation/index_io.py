#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Utilities for reading and writing cross-validation index mappings."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast


def serialize_sample(sample: Any) -> Dict[str, Any]:
    """Serialize a sample object to a JSON-friendly dict."""
    if hasattr(sample, "to_dict") and callable(sample.to_dict):
        return cast(Dict[str, Any], sample.to_dict())
    if isinstance(sample, dict):
        return sample
    raise TypeError(f"Unsupported sample type for serialization: {type(sample)}")


def fold_dir_for(index_mapping_dir: Path, strategy: str) -> Path:
    """Return the directory used for split fold JSON files."""
    return index_mapping_dir / f"camera_pairs_{strategy}_folds"


def existing_fold_files(index_mapping_dir: Path, strategy: str) -> List[Path]:
    """Return sorted existing fold files for a strategy."""
    fold_dir = fold_dir_for(index_mapping_dir, strategy)
    if not fold_dir.exists():
        return []
    return sorted(p.resolve() for p in fold_dir.glob("fold_*.json") if p.is_file())


def save_fold_files(
    folds: Dict[int, Dict[str, Any]],
    strategy: str,
    index_mapping_dir: Path,
) -> List[Path]:
    """Save each fold into an individual JSON file and return written paths."""
    fold_dir = fold_dir_for(index_mapping_dir, strategy)
    fold_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []
    for fold_idx in sorted(folds.keys()):
        fold_data = folds[fold_idx]

        serialized_fold: Dict[str, Any] = {
            "train": [serialize_sample(s) for s in fold_data["train"]],
            "val": [serialize_sample(s) for s in fold_data["val"]],
            "test": [serialize_sample(s) for s in fold_data.get("test", [])],
        }

        for key, value in fold_data.items():
            if key not in ["train", "val", "test"]:
                serialized_fold[key] = value

        serialized_fold["_metadata"] = {
            "strategy": strategy,
            "fold_idx": int(fold_idx),
            "num_train": len(fold_data["train"]),
            "num_val": len(fold_data["val"]),
            "num_test": len(fold_data.get("test", [])),
            "total": (
                len(fold_data["train"])
                + len(fold_data["val"])
                + len(fold_data.get("test", []))
            ),
        }

        fold_file = fold_dir / f"fold_{int(fold_idx):02d}.json"
        with fold_file.open("w", encoding="utf-8") as f:
            json.dump(serialized_fold, f, ensure_ascii=False, indent=2)

        saved_files.append(fold_file.resolve())

    return saved_files


def remove_legacy_aggregate_file(index_mapping_dir: Path, strategy: str) -> Optional[Path]:
    """Remove the old aggregate JSON file if it exists."""
    aggregate_file = index_mapping_dir / f"camera_pairs_{strategy}.json"
    if not aggregate_file.exists():
        return None
    aggregate_file.unlink()
    return aggregate_file
