"""Shared target-building helpers for equipment trainers."""

from typing import List

import torch


def select_points(x: torch.Tensor, idx: List[int], name: str) -> torch.Tensor:
    """Select points from [B,J,3] or [B,T,J,3] tensors."""
    if x.ndim == 3:
        point_dim = 1
    elif x.ndim == 4:
        point_dim = 2
    else:
        raise ValueError(f"Unexpected {name} ndim: {x.ndim}")

    if x.shape[-1] != 3:
        raise ValueError(f"Expected {name} last dimension 3, got {tuple(x.shape)}")

    max_idx = x.shape[point_dim] - 1
    if any(i < 0 or i > max_idx for i in idx):
        raise ValueError(f"Invalid {name} index in {idx}, valid range is [0, {max_idx}]")
    return x.index_select(point_dim, torch.as_tensor(idx, device=x.device))


def build_object_gt(
    pole_gt: torch.Tensor,
    ski_gt: torch.Tensor,
    pole_idx: List[int],
    ski_idx: List[int],
) -> torch.Tensor:
    """Build canonical equipment target [..., 4, 2, 3] from Unity GT tensors."""
    ski_obj = select_points(ski_gt, ski_idx, "ski_gt")
    pole_obj = select_points(pole_gt, pole_idx, "pole_gt")
    return torch.stack([ski_obj, pole_obj], dim=-2)



def build_anchor_points(
    human_3d: torch.Tensor,
    left_foot_idx: int = 10,
    right_foot_idx: int = 11,
    left_hand_idx: int = 13,
    right_hand_idx: int = 12,
) -> torch.Tensor:
    """Build per-equipment body anchors with shape [..., 4, 2, 3].

    Anchor order matches object targets: left ski, right ski, left pole, right pole.
    Both endpoints of each equipment segment use the same body anchor, so models can
    learn local offsets instead of absolute scene coordinates.
    """
    if human_3d.ndim not in (3, 4):
        raise ValueError(f"Expected human_3d shape [B,J,3] or [B,T,J,3], got {tuple(human_3d.shape)}")
    if human_3d.shape[-1] != 3:
        raise ValueError(f"Expected human_3d last dimension 3, got {tuple(human_3d.shape)}")

    point_dim = human_3d.ndim - 2
    max_idx = human_3d.shape[point_dim] - 1
    idx = [left_foot_idx, right_foot_idx, left_hand_idx, right_hand_idx]
    if any(i < 0 or i > max_idx for i in idx):
        raise ValueError(f"Invalid anchor index in {idx}, valid range is [0, {max_idx}]")

    anchor_idx = torch.as_tensor(idx, device=human_3d.device)
    anchors = human_3d.index_select(point_dim, anchor_idx)
    return anchors.unsqueeze(-2).expand(*anchors.shape[:-1], 2, 3)
