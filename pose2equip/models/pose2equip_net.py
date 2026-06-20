"""Backward-compatible exports for Pose2Equip model components."""

import torch
import torch.nn as nn

from pose2equip.models.equipment_decoder import EquipmentQueryDecoder
from pose2equip.models.image_encoder import DinoPatchEncoder
from pose2equip.models.pose2equip_model import Pose2EquipNet
from pose2equip.models.pose_encoder import PoseEncoder
from pose2equip.models.stgcn_baseline import STGCNBaselineNet


class DynamicQueryInit(nn.Module):
    """Compatibility helper for older experiments."""

    def __init__(self, key_joint_idx=None, hidden_dim=256):
        super().__init__()
        self.key_joint_idx = list(key_joint_idx or [])
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pose_context: torch.Tensor, joint_pos: torch.Tensor):
        if pose_context.ndim == 4:
            context = pose_context.mean(dim=2)
        elif pose_context.ndim == 3:
            context = pose_context
        else:
            raise ValueError(f"Unexpected pose_context shape {tuple(pose_context.shape)}")

        if joint_pos.ndim == 4:
            anchor = joint_pos.mean(dim=2)
        elif joint_pos.ndim == 3:
            anchor = joint_pos
        else:
            raise ValueError(f"Unexpected joint_pos shape {tuple(joint_pos.shape)}")

        return self.proj(torch.cat([context, anchor], dim=-1)).unsqueeze(2)


class ImprovedEquipmentQueryDecoder(EquipmentQueryDecoder):
    """Backward-compatible name for the current equipment query decoder."""


class Pose2EquipNetImproved(Pose2EquipNet):
    """Backward-compatible name for the current Pose2EquipNet implementation."""


class STGCNBaselineNetImproved(STGCNBaselineNet):
    """Backward-compatible name for the current STGCN baseline implementation."""


__all__ = [
    "DinoPatchEncoder",
    "PoseEncoder",
    "EquipmentQueryDecoder",
    "Pose2EquipNet",
    "STGCNBaselineNet",
    "DynamicQueryInit",
    "ImprovedEquipmentQueryDecoder",
    "Pose2EquipNetImproved",
    "STGCNBaselineNetImproved",
]
