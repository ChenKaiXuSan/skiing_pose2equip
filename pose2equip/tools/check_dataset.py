#!/usr/bin/env python3
"""Check Unity fold mappings before launching training."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf

from pose2equip.data_index import (
    fold_file_for,
    load_fold_dataset_idx_from_fold_json,
)
from pose2equip.dataloader.unity_dataset_single_view import single_view_dataset
from pose2equip.map_config import UnityDataConfig


def load_config(path: Path):
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def sample_indices(length: int, limit: int) -> Iterable[int]:
    if limit <= 0 or length == 0:
        return []
    return range(min(length, limit))


def summarize_sample(sample: Dict[str, Any]) -> str:
    parts = []
    if "frame_indices" in sample:
        parts.append(f"frames={tuple(sample['frame_indices'].shape)}")
    if "kpt3d_gt" in sample:
        shapes = {k: tuple(v.shape) for k, v in sample["kpt3d_gt"].items()}
        parts.append(f"kpt3d_gt={shapes}")
    if "frames" in sample and isinstance(sample["frames"], dict):
        shapes = {k: tuple(v.shape) for k, v in sample["frames"].items()}
        parts.append(f"images={shapes}")
    return ", ".join(parts) or "sample loaded"


def check_split(
    split: str,
    items: List[UnityDataConfig],
    *,
    experiment: str,
    num_samples: int,
    target_t: int,
    load_frames: bool,
    load_2d_kpt: bool,
    load_3d_kpt: bool,
) -> Dict[str, Any]:
    result = {
        "split": split,
        "total": len(items),
        "checked": 0,
        "ok": 0,
        "failed": 0,
        "errors": [],
    }
    if len(items) == 0:
        return result

    dataset = single_view_dataset(
        experiment=experiment,
        dataset_idx=items,
        transform=None,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
        target_t=target_t,
    )

    for idx in sample_indices(len(dataset), num_samples):
        result["checked"] += 1
        try:
            sample = dataset[idx]
            result["ok"] += 1
            print(f"[{split} #{idx}] OK: {summarize_sample(sample)}")
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic CLI.
            result["failed"] += 1
            error = f"{type(exc).__name__}: {exc}"
            result["errors"].append({"index": idx, "error": error})
            print(f"[{split} #{idx}] FAIL: {error}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pose2equip.yaml"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--index-mapping-path", type=Path, default=None)
    parser.add_argument("--unity-root", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--target-t", type=int, default=None)
    parser.add_argument("--load-frames", action="store_true")
    parser.add_argument("--load-2d-kpt", action="store_true")
    parser.add_argument("--no-load-3d-kpt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    unity_root = args.unity_root or Path(str(cfg.data.unity.root_path))
    index_mapping_path = args.index_mapping_path or Path(str(cfg.data.index_mapping_path))
    fold_file = fold_file_for(index_mapping_path, args.fold)
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")

    target_t = args.target_t if args.target_t is not None else int(cfg.data.time_window)
    load_3d_kpt = not args.no_load_3d_kpt
    if not args.load_frames and not args.load_2d_kpt and not load_3d_kpt:
        raise ValueError("At least one modality must be enabled.")

    print(f"Config: {args.config}")
    print(f"Unity root: {unity_root}")
    print(f"Fold file: {fold_file}")
    print(
        f"Checking splits={args.splits}, "
        f"num_samples={args.num_samples}, target_t={target_t}"
    )

    fold = load_fold_dataset_idx_from_fold_json(
        cfg,
        args.fold,
        index_mapping_path=index_mapping_path,
        unity_root=unity_root,
    )
    results = []
    for split in args.splits:
        results.append(
            check_split(
                split,
                fold[split],
                experiment=str(cfg.experiment),
                num_samples=args.num_samples,
                target_t=target_t,
                load_frames=args.load_frames,
                load_2d_kpt=args.load_2d_kpt,
                load_3d_kpt=load_3d_kpt,
            )
        )

    total_failed = sum(r["failed"] for r in results)
    print("\nSummary:")
    for r in results:
        print(
            f"  {r['split']}: total={r['total']} checked={r['checked']} "
            f"ok={r['ok']} failed={r['failed']}"
        )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
