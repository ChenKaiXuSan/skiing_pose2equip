"""Pose encoders for 3D human keypoint sequences."""

import torch.nn as nn

from pose2equip.models.stgcn import STGCN


class PoseEncoder(nn.Module):
    def __init__(
        self,
        num_joints=17,
        hidden_dim=256,
        target_skeleton_connections_idx=None,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.stgcn = STGCN(
            num_joints=self.num_joints,
            in_channels=3,
            hidden_channels=(64, 64, 128, 128, hidden_dim),
            edges=target_skeleton_connections_idx,
            dropout=0.1,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, human_3d):
        """Encode human_3d [B,T,J,3] as pose memory [B,T,J,C]."""
        feat, _ = self.stgcn(human_3d, return_features=True)
        return self.proj(feat)
