#!/usr/bin/env python3
"""Behavior tests for equipment geometry helpers."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pose2equip.losses.equipment import temporal_smoothness_loss
from pose2equip.trainer.equipment_targets import build_anchor_points


def test_build_anchor_points_matches_feet_and_hands():
    human_3d = torch.zeros(2, 3, 15, 3)
    human_3d[:, :, 10] = torch.tensor([1.0, 0.0, 0.0])
    human_3d[:, :, 11] = torch.tensor([2.0, 0.0, 0.0])
    human_3d[:, :, 13] = torch.tensor([3.0, 0.0, 0.0])
    human_3d[:, :, 12] = torch.tensor([4.0, 0.0, 0.0])

    anchors = build_anchor_points(human_3d)

    assert anchors.shape == (2, 3, 4, 2, 3)
    assert torch.allclose(anchors[:, :, 0], human_3d[:, :, 10].unsqueeze(2).expand(-1, -1, 2, -1))
    assert torch.allclose(anchors[:, :, 1], human_3d[:, :, 11].unsqueeze(2).expand(-1, -1, 2, -1))
    assert torch.allclose(anchors[:, :, 2], human_3d[:, :, 13].unsqueeze(2).expand(-1, -1, 2, -1))
    assert torch.allclose(anchors[:, :, 3], human_3d[:, :, 12].unsqueeze(2).expand(-1, -1, 2, -1))


def test_temporal_smoothness_loss_penalizes_frame_jumps():
    static_obj = torch.zeros(2, 4, 4, 2, 3)
    jump_obj = static_obj.clone()
    jump_obj[:, 2:] = 10.0

    assert temporal_smoothness_loss(static_obj).item() == 0.0
    assert temporal_smoothness_loss(jump_obj).item() > 0.0


if __name__ == "__main__":
    test_build_anchor_points_matches_feet_and_hands()
    test_temporal_smoothness_loss_penalizes_frame_jumps()
    print("equipment geometry tests passed")
