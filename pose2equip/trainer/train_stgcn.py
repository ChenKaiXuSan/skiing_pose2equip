from typing import Dict

import torch

from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS
from pose2equip.models.stgcn_baseline import STGCNBaselineNet
from pose2equip.trainer.base_equipment_trainer import BaseEquipmentTrainer


class Pose2Equip_STGCN_Trainer(BaseEquipmentTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = STGCNBaselineNet(
            num_joints=15,
            hidden_dim=int(getattr(args.pose2equip, "hidden_dim", 256)),
            num_equip_kpts=8,
            target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
        )

    def forward_model(
        self, human_3d: torch.Tensor, human_frame: torch.Tensor | None
    ) -> Dict[str, torch.Tensor]:
        return self.model(human_3d)
