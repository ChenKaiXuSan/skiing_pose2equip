"""Multimodal Pose2Equip network."""

import torch
import torch.nn as nn

from pose2equip.models.equipment_decoder import EquipmentQueryDecoder
from pose2equip.models.image_encoder import DinoPatchEncoder
from pose2equip.models.pose_encoder import PoseEncoder


class Pose2EquipNet(nn.Module):
    """RGB sequence + 3D human pose -> equipment endpoint offsets."""

    def __init__(
        self,
        num_joints=17,
        hidden_dim=256,
        target_skeleton_connections_idx=None,
        dino_model_name="facebook/dinov2-base",
        dino_freeze=True,
        decoder_layers=3,
        num_heads=8,
    ):
        super().__init__()
        self.image_encoder = DinoPatchEncoder(
            model_name=dino_model_name,
            out_dim=hidden_dim,
            freeze=dino_freeze,
        )
        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )
        self.decoder = EquipmentQueryDecoder(
            num_queries=4,
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
        )

    def forward(self, human_frame: torch.Tensor, human_3d: torch.Tensor):
        if human_frame is None:
            raise ValueError(
                "Pose2EquipNet requires human_frame input with shape [B,T,3,H,W]."
            )
        image_tokens = self.image_encoder(human_frame)
        pose_tokens = self.pose_encoder(human_3d)
        memory = torch.cat([image_tokens, pose_tokens], dim=2)
        return {"object_3d": self.decoder(memory)}
