#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from pose2equip.dataloader.data_loader import UnityDataModule
from pose2equip.data_index import (
    detect_available_folds,
    load_fold_dataset_idx_from_fold_json,
)

logger = logging.getLogger(__name__)


def resolve_trainer_device_kwargs(hparams: DictConfig) -> dict:
    """Resolve Lightning accelerator/devices from config with CPU-friendly defaults."""
    accelerator = str(getattr(hparams.trainer, "accelerator", "auto"))
    devices = getattr(hparams.trainer, "devices", "auto")
    if devices is None:
        devices = "auto"
    return {"accelerator": accelerator, "devices": devices}


def resolve_test_ckpt_path(hparams: DictConfig):
    """Return which checkpoint Lightning should use for test after fit."""
    return getattr(hparams.trainer, "test_ckpt_path", "best")


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    monitor_metric = "val/video_acc"
    monitor_mode = "max"
    ckpt_filename = "{epoch}-{val/loss:.2f}-{val/video_acc:.4f}"

    if hparams.model.backbone == "pose2equip":
        from .trainer.train_pose2equip import Pose2EquipTrainer

        lit_module = Pose2EquipTrainer(hparams)
        # pose2equip 当前验证阶段记录的是 val/mpjpe 与 val/loss。
        monitor_metric = "val/mpjpe"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}"
    elif hparams.model.backbone == "stgcn":
        from .trainer.train_stgcn import Pose2Equip_STGCN_Trainer

        lit_module = Pose2Equip_STGCN_Trainer(hparams)
        # stgcn 当前验证阶段记录的是 val/mpjpe 与 val/loss。
        monitor_metric = "val/mpjpe"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}"
    elif hparams.model.backbone in {"mlp", "tcn", "skeleton_transformer", "stgcn_query"}:
        from .trainer.train_skeleton_backbone import SkeletonBackboneTrainer

        lit_module = SkeletonBackboneTrainer(hparams)
        monitor_metric = "val/mpjpe"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}"
    elif hparams.model.backbone == "3dcnn":
        from .trainer.train_3dcnn import Res3DCNNTrainer

        lit_module = Res3DCNNTrainer(hparams)
    else:
        raise ValueError(
            f"Unsupported model.backbone={hparams.model.backbone}. "
            "Expected one of: pose2equip, stgcn, mlp, tcn, skeleton_transformer, stgcn_query, 3dcnn."
        )

    # * prepare data module
    data_module = UnityDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=10)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", "fold_" + str(fold)),
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=2,
    )

    # # define the early stop.
    # early_stopping = EarlyStopping(
    #     monitor=monitor_metric,
    #     patience=5,
    #     mode=monitor_mode,
    # )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        **resolve_trainer_device_kwargs(hparams),
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            # early_stopping,
            lr_monitor,
        ],
        limit_train_batches=getattr(hparams.trainer, "limit_train_batches", None),
        limit_val_batches=getattr(hparams.trainer, "limit_val_batches", None),
        limit_test_batches=getattr(hparams.trainer, "limit_test_batches", None),
        num_sanity_val_steps=int(getattr(hparams.trainer, "num_sanity_val_steps", 2)),
    )

    trainer.fit(lit_module, data_module)

    # save the metrics to file
    trainer.test(
        lit_module,
        data_module,
        ckpt_path=resolve_test_ckpt_path(hparams),
        weights_only=False,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="pose2equip.yaml",
)
def init_params(config):

    # Load precomputed fold mapping only; do not prepare CV splits here.
    # 使用预生成的单fold JSON文件（每个fold文件必须存在）

    requested_fold = int(config.train.fold)

    # 检测可用的fold数量
    available_folds = detect_available_folds(config)

    # train.fold >= 0: run only the specified fold (recommended for multi-node jobs)
    # train.fold < 0: run all folds sequentially (backward compatible mode)
    if requested_fold >= 0:
        if requested_fold not in available_folds:
            raise KeyError(
                f"Requested fold {requested_fold} is not available. "
                f"Available folds: {available_folds}"
            )
        target_folds = [requested_fold]
    else:
        target_folds = available_folds

    logger.info("#" * 50)
    logger.info(
        "Start training folds: %s (requested train.fold=%s)",
        target_folds,
        requested_fold,
    )
    logger.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold in target_folds:
        # 加载单个fold的JSON文件
        dataset_value = load_fold_dataset_idx_from_fold_json(config, fold)
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train folds: %s", target_folds)
    logger.info("#" * 50)



if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
