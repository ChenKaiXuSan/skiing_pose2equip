from typing import Any, Dict

import torch

from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS
from pose2equip.models.pose2equip_model import Pose2EquipNet
from pose2equip.trainer.base_equipment_trainer import BaseEquipmentTrainer


class Pose2EquipTrainer(BaseEquipmentTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
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

    def _extract_human_frame(self, batch: Dict[str, Any]) -> torch.Tensor | None:
        if isinstance(batch.get("frames"), dict) and "cam1" in batch["frames"]:
            human_frame = batch["frames"]["cam1"].float()
            return human_frame.permute(0, 2, 1, 3, 4).contiguous()
        return None

    def forward_model(
        self, human_3d: torch.Tensor, human_frame: torch.Tensor | None
    ) -> Dict[str, torch.Tensor]:
        return self.model(human_3d=human_3d, human_frame=human_frame)
