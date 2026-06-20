from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from pose2equip.trainer.base_equipment_trainer import BaseEquipmentTrainer


class DummyEquipmentTrainer(BaseEquipmentTrainer):
    def forward_model(self, human_3d, human_frame):
        return {"object_3d": torch.zeros(*human_3d.shape[:2], 4, 2, 3)}


def make_args(source="unity", sam_key="character_cam1"):
    return SimpleNamespace(
        data=SimpleNamespace(human_3d_source=source, sam3d_human_key=sam_key),
        loss=SimpleNamespace(lr=0.001, weight_decay=0.01),
        pose2equip=SimpleNamespace(
            loss_w_sym=0.03,
            loss_w_len_abs=0.2,
            loss_w_temporal_smooth=0.02,
            predict_anchor_offsets=False,
            ski_gt_idx=[1, 2, 4, 5],
            pole_gt_idx=[0, 1, 2, 3],
            left_foot_idx=10,
            right_foot_idx=11,
            left_hand_idx=13,
            right_hand_idx=12,
        ),
        log_path="/tmp/pose2equip-test",
    )


def make_batch():
    unity = torch.ones(2, 3, 15, 3)
    sam = torch.full((2, 3, 15, 3), 7.0)
    return {
        "kpt3d_gt": {
            "character": unity,
            "pole": torch.zeros(2, 3, 4, 3),
            "ski": torch.zeros(2, 3, 6, 3),
        },
        "kpt3d_sam": {"character_cam1": sam},
    }


def test_select_human_3d_uses_unity_source():
    trainer = DummyEquipmentTrainer(make_args(source="unity"))

    human_3d = trainer._select_human_3d(make_batch())

    assert torch.equal(human_3d, torch.ones(2, 3, 15, 3))


def test_select_human_3d_uses_sam3d_source():
    trainer = DummyEquipmentTrainer(make_args(source="sam3d"))

    human_3d = trainer._select_human_3d(make_batch())

    assert torch.equal(human_3d, torch.full((2, 3, 15, 3), 7.0))


def test_pose2equip_config_declares_human_3d_source_defaults():
    cfg = OmegaConf.load("configs/pose2equip.yaml")

    assert cfg.data.human_3d_source == "unity"
    assert cfg.data.sam3d_human_key == "character_cam1"
