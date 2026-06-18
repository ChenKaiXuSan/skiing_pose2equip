from pathlib import Path
from typing import Any, Dict, List
import logging

import torch
from pytorch_lightning import (
    LightningModule,
)
from pose2equip.losses.equipment import (
    absolute_length_loss,
    mpjpe,
    symmetry_loss,
    temporal_smoothness_loss,
)
from pose2equip.metrics.equipment import evaluate_pose_metrics
from pose2equip.models.pose2equip_net import Pose2EquipNet
from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS
from pose2equip.trainer.equipment_targets import build_anchor_points, build_object_gt

logger = logging.getLogger(__name__)


class Pose2EquipTrainer(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = Pose2EquipNet(
            num_joints=15,
            target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
            dino_model_name=str(
                getattr(
                    args.pose2equip,
                    "dino_model_name",
                    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
                )
            ),
            dino_freeze=bool(getattr(args.pose2equip, "dino_freeze", True)),
        )
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

        # GT point reorder for object_3d target (8 points):
        # [left_ski_tip, left_ski_tail, right_ski_tip, right_ski_tail,
        #  left_pole_grip, left_pole_tip, right_pole_grip, right_pole_tip]
        # 这个里面应该和unity记录的GT点顺序一致，或者至少保证能正确选取对应的点进行训练
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

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        human_frame = None
        if isinstance(batch.get("frames"), dict) and "cam1" in batch["frames"]:
            human_frame = batch["frames"]["cam1"].float()  # [B, C, T, H, W]
            human_frame = human_frame.permute(
                0, 2, 1, 3, 4
            ).contiguous()  # [B, T, C, H, W]

        # GT fron Unity
        _gt = batch["kpt3d_gt"]
        pole_gt = _gt["pole"].float()  # [B, t, 4, 3]
        ski_gt = _gt["ski"].float()  # [B, t, 6, 3]
        human_3d_gt = _gt["character"].float()  # [B, t, J, 3]

        object_gt = self._build_object_gt(
            pole_gt=pole_gt, ski_gt=ski_gt
        )  # [B, T, 4, 2, 3]

        out = self.model(
            human_3d=human_3d_gt,
            human_frame=human_frame,
        )
        pred_offset = out["object_3d"]
        pred_obj = self._restore_anchor_offsets(pred_offset, human_3d_gt)

        l3d = mpjpe(pred_obj, object_gt)
        lsymmetry = symmetry_loss(pred_obj)
        l_len_abs = absolute_length_loss(pred_obj=pred_obj, gt_obj=object_gt)
        l_temporal = temporal_smoothness_loss(pred_obj)

        # Final objective:
        #   L = L3D + w_sym * Lsymmetry + w_len_abs * LlenAbs + w_tmp * Ltemporal
        # L3D is the main supervision term; others are geometric regularizers.
        loss = (
            l3d
            + self.loss_w_sym * lsymmetry
            + self.loss_w_len_abs * l_len_abs
            + self.loss_w_temporal_smooth * l_temporal
        )

        batch_size = human_3d_gt.shape[0]
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/mpjpe",
            l3d,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/L3D",
            l3d,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Lsymmetry",
            lsymmetry,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Llen_abs",
            l_len_abs,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/Ltemporal",
            l_temporal,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        if stage == "test":
            self.test_outputs.append(
                {
                    "human_3d": human_3d_gt.detach().cpu(),
                    "pred_obj": pred_obj.detach().cpu(),
                    "gt_obj": object_gt.detach().cpu(),
                }
            )

        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def test_step(
        self, batch: Dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def on_test_epoch_end(self) -> None:
        if len(self.test_outputs) == 0:
            return

        payload = {
            "human_3d": torch.cat([x["human_3d"] for x in self.test_outputs], dim=0),
            "pred_obj": torch.cat([x["pred_obj"] for x in self.test_outputs], dim=0),
            "gt_obj": torch.cat([x["gt_obj"] for x in self.test_outputs], dim=0),
        }
        save_file = self.test_save_dir / "pose2equip_outputs.pt"
        torch.save(payload, save_file)

        # Compute performance metrics
        pred_obj_np = payload["pred_obj"].numpy()
        gt_obj_np = payload["gt_obj"].numpy()
        metrics = evaluate_pose_metrics(pred_obj_np, gt_obj_np)

        # Save metrics to txt
        metrics_file = self.test_save_dir / "evaluation_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Equipment 3D Keypoint Prediction - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total samples evaluated: {pred_obj_np.shape[0]}\n\n")

            f.write("Global Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(
                f"  MPJPE (Mean Per Joint Position Error):  {metrics['mpjpe']:.4f} mm\n"
            )
            f.write(
                f"  PA-MPJPE (Procrustes Aligned):         {metrics['pa_mpjpe']:.4f} mm\n\n"
            )

            f.write("Per-Object Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Left Ski MPJPE:   {metrics['mpjpe_left_ski']:.4f} mm\n")
            f.write(f"  Right Ski MPJPE:  {metrics['mpjpe_right_ski']:.4f} mm\n")
            f.write(f"  Left Pole MPJPE:  {metrics['mpjpe_left_pole']:.4f} mm\n")
            f.write(f"  Right Pole MPJPE: {metrics['mpjpe_right_pole']:.4f} mm\n\n")

            ski_avg = (metrics["mpjpe_left_ski"] + metrics["mpjpe_right_ski"]) / 2.0
            pole_avg = (metrics["mpjpe_left_pole"] + metrics["mpjpe_right_pole"]) / 2.0
            f.write(f"  Avg Ski Error:    {ski_avg:.4f} mm\n")
            f.write(f"  Avg Pole Error:   {pole_avg:.4f} mm\n\n")

            f.write("=" * 60 + "\n")

        logger.info(f"Evaluation metrics saved to {metrics_file}")
        logger.info(
            f"MPJPE: {metrics['mpjpe']:.4f} mm, PA-MPJPE: {metrics['pa_mpjpe']:.4f} mm"
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
