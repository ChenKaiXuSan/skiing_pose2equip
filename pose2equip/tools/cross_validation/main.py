#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
生成交叉验证索引文件
用于保存摄像头两两组合的交叉验证划分结果
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf


from pose2equip.tools.cross_validation.cross_validation_camera_pairs import CameraPairCrossValidation
from pose2equip.tools.cross_validation.index_io import (
    existing_fold_files,
    fold_dir_for,
    remove_legacy_aggregate_file,
    save_fold_files,
)


def generate_index_files(
    data_root: str,
    num_persons: int = 2,
    num_actions: int = 12,
    num_cameras: int = 108,
    use_layer_camera_filter: bool = False,
    selected_layers: Optional[List[int]] = None,
    selected_cameras_per_layer: Optional[Dict[str, List[int]]] = None,
    strategies: Optional[List[str]] = None,
    n_splits: int = 5,
    force_recreate: bool = False,
    sam3d_export_root: Optional[str] = None,
):
    """
    生成所有策略的索引文件

    Args:
        data_root: 数据根目录
        num_persons: 人物数量
        num_actions: 动作数量
        num_cameras: 摄像头数量
        use_layer_camera_filter: 是否启用按层/层内相机筛选
        selected_layers: 选中的层列表
        selected_cameras_per_layer: 每层选中的相机编号
        strategies: 要生成的策略列表，默认生成 by_action 和 by_camera_pair
        n_splits: K折数量
        force_recreate: 是否强制重新生成
    """
    if strategies is None:
        strategies = ["by_action", "by_camera_pair"]

    index_mapping_dir = (
        Path(data_root)
        / "index_mapping"
        / f"use_layer_camera_filter_{'enabled' if use_layer_camera_filter else 'disabled'}"
    )
    index_mapping_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("生成交叉验证索引文件")
    print("=" * 80)
    print(f"数据根目录: {data_root}")
    print(f"人物数: {num_persons}")
    print(f"动作数: {num_actions}")
    print(f"摄像头数: {num_cameras}")
    print(
        "层/相机筛选: "
        f"{use_layer_camera_filter} (layers={selected_layers or []}, "
        f"cameras_per_layer={selected_cameras_per_layer or {}})"
    )
    print(f"策略: {', '.join(strategies)}")
    print(f"保存目录: {index_mapping_dir}")
    print("=" * 80 + "\n")

    results: Dict[str, Dict[str, Any]] = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"生成策略: {strategy}")
        print(f"{'='*60}")

        # 创建交叉验证对象
        cv = CameraPairCrossValidation(
            data_root=data_root,
            num_persons=num_persons,
            num_actions=num_actions,
            num_cameras=num_cameras,
            use_layer_camera_filter=use_layer_camera_filter,
            selected_layers=selected_layers,
            selected_cameras_per_layer=selected_cameras_per_layer,
            split_strategy=strategy,
            n_splits=n_splits,
            sam3d_export_root=sam3d_export_root,
            index_save_path=str(index_mapping_dir / f"camera_pairs_{strategy}.json"),
        )

        # 仅生成 fold 划分文件，不再保存整体聚合 json。
        fold_dir = fold_dir_for(index_mapping_dir, strategy)
        if not force_recreate:
            fold_files = existing_fold_files(index_mapping_dir, strategy)
            if fold_files:
                print("✓ 发现已存在的 fold 划分文件，直接复用")
                results[strategy] = {
                    "fold_dir": str(fold_dir.resolve()),
                    "fold_files": fold_files,
                    "n_folds": len(fold_files),
                    "total_samples": "unknown",
                }
                print(f"  Fold目录: {results[strategy]['fold_dir']}")
                print(f"  Fold文件数: {len(fold_files)}")
                continue

        folds = cv.prepare_folds()
        fold_files = save_fold_files(
            folds=folds,
            strategy=strategy,
            index_mapping_dir=index_mapping_dir,
        )
        removed_file = remove_legacy_aggregate_file(index_mapping_dir, strategy)
        if removed_file is not None:
            print(f"  已删除旧的整体索引文件: {removed_file}")

        # 记录结果
        strategy_fold_dir = (
            index_mapping_dir / f"camera_pairs_{strategy}_folds"
        ).resolve()

        results[strategy] = {
            "fold_dir": str(strategy_fold_dir),
            "fold_files": fold_files,
            "n_folds": len(folds),
            "total_samples": sum(
                len(fold["train"]) + len(fold["val"]) + len(fold.get("test", []))
                for fold in folds.values()
            )
            // len(folds),
        }

        print(f"\n✓ 策略 '{strategy}' fold 索引文件已生成")
        print(f"  Fold目录: {results[strategy]['fold_dir']}")
        print(f"  Fold文件数: {len(fold_files)}")
        for fold_file in fold_files:
            print(f"    - {fold_file}")
        print(f"  折数: {results[strategy]['n_folds']}")
        print(f"  每折样本数: {results[strategy]['total_samples']}")

    # 打印汇总信息
    print("\n" + "=" * 80)
    print("✅ 所有索引文件生成完成")
    print("=" * 80)

    for strategy, info in results.items():
        fold_files_summary = cast(List[Path], info.get("fold_files", []))
        print(f"\n[{strategy}]")
        print(f"  Fold目录: {info['fold_dir']}")
        print(f"  Fold文件数: {len(fold_files_summary)}")
        print(f"  折数: {info['n_folds']}")
        total_samples = info["total_samples"]
        if isinstance(total_samples, int):
            print(f"  样本数: {total_samples:,}")

    print("\n" + "=" * 80)
    print("📁 所有索引文件位置:")
    print(f"   {index_mapping_dir}")
    print("=" * 80 + "\n")

    return results


@hydra.main(version_base=None, config_path="../../../configs", config_name="cross_validation.yaml")
def hydra_main(cfg: Optional[DictConfig] = None) -> None:
    """
    通过 Hydra 配置生成 index mapping。

    默认读取 configs/cross_validation.yaml，
    也支持命令行 override，例如：
    python -m pose2equip.tools.cross_validation.main cross_validation.force_recreate=true
    """
    if cfg is None:
        raise ValueError("Hydra cfg is required")

    cv_cfg = cfg.cross_validation

    # 打印当前配置，便于排查。
    print("\nHydra cross validation config:")
    print(OmegaConf.to_yaml(cv_cfg))

    generate_index_files(
        data_root=str(cfg.data.unity.root_path),
        num_persons=int(cv_cfg.num_persons),
        num_actions=int(cv_cfg.num_actions),
        num_cameras=int(cv_cfg.num_cameras),
        use_layer_camera_filter=bool(cv_cfg.get("use_layer_camera_filter", False)),
        selected_layers=[
            int(x) for x in cast(List[Any], list(cv_cfg.get("selected_layers", [])))
        ],
        selected_cameras_per_layer={
            str(layer): [int(cam_id) for cam_id in cast(List[Any], cam_ids)]
            for layer, cam_ids in cast(
                Dict[str, Any],
                OmegaConf.to_container(
                    cv_cfg.get("selected_cameras_per_layer", {}),
                    resolve=True,
                ),
            ).items()
        },
        strategies=list(cv_cfg.strategies),
        n_splits=int(cv_cfg.n_splits),
        force_recreate=bool(cv_cfg.force_recreate),
        sam3d_export_root=str(cfg.data.get("sam3d_export_path", "")),
    )


if __name__ == "__main__":
    cast(Any, hydra_main)()
