"""Trainer wrapper for pose-only skeleton comparison backbones."""

from typing import Dict

import torch

from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS
from pose2equip.models.skeleton_backbones import build_skeleton_backbone
from pose2equip.trainer.base_equipment_trainer import BaseEquipmentTrainer


class SkeletonBackboneTrainer(BaseEquipmentTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        backbone_name = str(args.model.backbone)
        self.model = build_skeleton_backbone(
            name=backbone_name,
            num_joints=int(getattr(args.pose2equip, "num_joints", 15)),
            hidden_dim=int(getattr(args.pose2equip, "hidden_dim", 256)),
            num_equip_kpts=int(getattr(args.pose2equip, "num_equip_kpts", 8)),
            target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
            num_layers=int(getattr(args.pose2equip, "backbone_layers", 3)),
            num_heads=int(getattr(args.pose2equip, "num_heads", 8)),
            dropout=float(getattr(args.pose2equip, "dropout", 0.1)),
            kernel_size=int(getattr(args.pose2equip, "tcn_kernel_size", 5)),
            decoder_layers=int(getattr(args.pose2equip, "decoder_layers", 3)),
        )

    def forward_model(
        self, human_3d: torch.Tensor, human_frame: torch.Tensor | None
    ) -> Dict[str, torch.Tensor]:
        return self.model(human_3d)
