"""Shared Lightning trainer logic for equipment keypoint models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from pose2equip.losses.equipment import (
    absolute_length_loss,
    mpjpe,
    symmetry_loss,
    temporal_smoothness_loss,
)
from pose2equip.metrics.equipment import evaluate_pose_metrics, save_evaluation_report
from pose2equip.trainer.equipment_targets import build_anchor_points, build_object_gt

logger = logging.getLogger(__name__)


class BaseEquipmentTrainer(LightningModule):
    """Common training, validation, and test logic for equipment predictors."""

    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(args.loss, "lr", 0.001))
        self.weight_decay = float(getattr(args.loss, "weight_decay", 0.01))

        self.loss_w_sym = float(getattr(args.pose2equip, "loss_w_sym", 0.03))
        self.loss_w_len_abs = float(getattr(args.pose2equip, "loss_w_len_abs", 0.2))
        self.loss_w_temporal_smooth = float(
            getattr(args.pose2equip, "loss_w_temporal_smooth", 0.02)
        )
        self.predict_anchor_offsets = bool(
            getattr(args.pose2equip, "predict_anchor_offsets", True)
        )
        self.human_3d_source = str(
            getattr(args.data, "human_3d_source", "unity")
        ).lower()
        self.sam3d_human_key = str(
            getattr(args.data, "sam3d_human_key", "character_cam1")
        )
        valid_human_sources = {"unity", "gt", "kpt3d_gt", "sam3d", "sam", "kpt3d_sam"}
        if self.human_3d_source not in valid_human_sources:
            raise ValueError(
                "data.human_3d_source must be one of "
                f"{sorted(valid_human_sources)}, got {self.human_3d_source!r}"
            )

        self.ski_gt_idx = list(getattr(args.pose2equip, "ski_gt_idx", [1, 2, 4, 5]))
        self.pole_gt_idx = list(getattr(args.pose2equip, "pole_gt_idx", [0, 1, 2, 3]))

        self.left_foot_idx = int(getattr(args.pose2equip, "left_foot_idx", 10))
        self.right_foot_idx = int(getattr(args.pose2equip, "right_foot_idx", 11))
        self.left_hand_idx = int(getattr(args.pose2equip, "left_hand_idx", 13))
        self.right_hand_idx = int(getattr(args.pose2equip, "right_hand_idx", 12))

        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir = Path(str(args.log_path)) / "pose_analysis"

    def _build_object_gt(
        self, pole_gt: torch.Tensor, ski_gt: torch.Tensor
    ) -> torch.Tensor:
        return build_object_gt(
            pole_gt=pole_gt,
            ski_gt=ski_gt,
            pole_idx=self.pole_gt_idx,
            ski_idx=self.ski_gt_idx,
        )

    def _restore_anchor_offsets(
        self, pred_offset: torch.Tensor, human_3d: torch.Tensor
    ) -> torch.Tensor:
        if not self.predict_anchor_offsets:
            return pred_offset
        anchors = build_anchor_points(
            human_3d,
            left_foot_idx=self.left_foot_idx,
            right_foot_idx=self.right_foot_idx,
            left_hand_idx=self.left_hand_idx,
            right_hand_idx=self.right_hand_idx,
        )
        return pred_offset + anchors

    def _extract_human_frame(self, batch: Dict[str, Any]) -> torch.Tensor | None:
        return None

    def _select_human_3d(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self.human_3d_source in {"unity", "gt", "kpt3d_gt"}:
            return batch["kpt3d_gt"]["character"].float()

        sam3d = batch.get("kpt3d_sam")
        if not isinstance(sam3d, dict):
            raise KeyError(
                "data.human_3d_source=sam3d requires batch['kpt3d_sam'] to be present."
            )
        if self.sam3d_human_key not in sam3d:
            raise KeyError(
                f"SAM3D human key {self.sam3d_human_key!r} not found in batch['kpt3d_sam']; "
                f"available keys: {sorted(sam3d.keys())}"
            )
        return sam3d[self.sam3d_human_key].float()

    def forward_model(
        self, human_3d: torch.Tensor, human_frame: torch.Tensor | None
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        human_frame = self._extract_human_frame(batch)

        gt = batch["kpt3d_gt"]
        pole_gt = gt["pole"].float()
        ski_gt = gt["ski"].float()
        human_3d = self._select_human_3d(batch)

        object_gt = self._build_object_gt(pole_gt=pole_gt, ski_gt=ski_gt)
        out = self.forward_model(human_3d=human_3d, human_frame=human_frame)
        pred_obj = self._restore_anchor_offsets(out["object_3d"], human_3d)

        l3d = mpjpe(pred_obj, object_gt)
        lsymmetry = symmetry_loss(pred_obj)
        l_len_abs = absolute_length_loss(pred_obj=pred_obj, gt_obj=object_gt)
        l_temporal = temporal_smoothness_loss(pred_obj)
        loss = (
            l3d
            + self.loss_w_sym * lsymmetry
            + self.loss_w_len_abs * l_len_abs
            + self.loss_w_temporal_smooth * l_temporal
        )

        batch_size = human_3d.shape[0]
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/mpjpe", l3d, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/L3D", l3d, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/Lsymmetry", lsymmetry, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/Llen_abs", l_len_abs, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/Ltemporal", l_temporal, on_step=True, on_epoch=True, batch_size=batch_size)

        if stage == "test":
            self.test_outputs.append(
                {
                    "human_3d": human_3d.detach().cpu(),
                    "pred_obj": pred_obj.detach().cpu(),
                    "gt_obj": object_gt.detach().cpu(),
                }
            )

        return loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def on_test_epoch_end(self) -> None:
        if len(self.test_outputs) == 0:
            return

        payload = {
            "human_3d": torch.cat([x["human_3d"] for x in self.test_outputs], dim=0),
            "pred_obj": torch.cat([x["pred_obj"] for x in self.test_outputs], dim=0),
            "gt_obj": torch.cat([x["gt_obj"] for x in self.test_outputs], dim=0),
        }
        torch.save(payload, self.test_save_dir / "pose2equip_outputs.pt")

        pred_obj_np = payload["pred_obj"].numpy()
        gt_obj_np = payload["gt_obj"].numpy()
        metrics = evaluate_pose_metrics(pred_obj_np, gt_obj_np)
        report_paths = save_evaluation_report(
            metrics=metrics,
            total_samples=pred_obj_np.shape[0],
            output_dir=self.test_save_dir,
        )

        logger.info(
            "Evaluation metrics saved to %s and %s",
            report_paths["txt"],
            report_paths["json"],
        )
        logger.info(
            "MPJPE: %.4f mm, PA-MPJPE: %.4f mm",
            metrics["mpjpe"],
            metrics["pa_mpjpe"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
