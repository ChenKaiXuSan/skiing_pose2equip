"""Pose-only ST-GCN equipment baseline."""

import torch
import torch.nn as nn

from pose2equip.models.pose_encoder import PoseEncoder


class STGCNBaselineNet(nn.Module):
    """STGCN-only baseline: 3D human keypoints -> equipment endpoint offsets."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        target_skeleton_connections_idx=None,
    ):
        super().__init__()
        self.num_equip_kpts = int(num_equip_kpts)
        if self.num_equip_kpts % 2 != 0:
            raise ValueError(
                f"num_equip_kpts must be even (pairs of endpoints), got {self.num_equip_kpts}"
            )
        self.num_equip = self.num_equip_kpts // 2
        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )
        self.equip_head = nn.Linear(hidden_dim, self.num_equip_kpts * 3)

    def forward(self, human_3d: torch.Tensor):
        if human_3d.ndim == 3:
            human_3d = human_3d.unsqueeze(1)
            single_frame_mode = True
        elif human_3d.ndim == 4:
            single_frame_mode = False
        else:
            raise ValueError(
                f"Expected human_3d shape [B,J,3] or [B,T,J,3], got {tuple(human_3d.shape)}"
            )

        b = human_3d.shape[0]
        t = human_3d.shape[1]
        pose_feat = self.pose_encoder(human_3d).mean(dim=2)
        pred_obj = self.equip_head(pose_feat).reshape(b, t, self.num_equip, 2, 3)
        if single_frame_mode:
            pred_obj = pred_obj.squeeze(1)
        return {"object_3d": pred_obj}
