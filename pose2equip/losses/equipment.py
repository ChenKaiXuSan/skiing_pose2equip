"""Loss functions for 3D equipment keypoint prediction."""

import torch


def to_equip_point_repr(obj: torch.Tensor) -> torch.Tensor:
    """Normalize object points to [..., 4, 2, 3]."""
    if obj.ndim >= 3 and tuple(obj.shape[-3:]) == (4, 2, 3):
        return obj
    if obj.ndim >= 2 and tuple(obj.shape[-2:]) == (8, 3):
        return obj.reshape(*obj.shape[:-2], 4, 2, 3)
    raise ValueError(
        f"Expected object shape ending with [4,2,3] or [8,3], got {tuple(obj.shape)}"
    )


def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean per-point Euclidean distance on equipment keypoints."""
    pred = to_equip_point_repr(pred)
    gt = to_equip_point_repr(gt)
    return torch.norm(pred - gt, dim=-1).mean()


def equipment_segment_lengths(obj: torch.Tensor) -> torch.Tensor:
    """Return segment lengths in order: left ski, right ski, left pole, right pole."""
    obj = to_equip_point_repr(obj)
    return torch.norm(obj[..., 0, :] - obj[..., 1, :], dim=-1)


def length_variance_loss(pred_obj: torch.Tensor) -> torch.Tensor:
    """Penalize sample-wise equipment length jitter in a batch."""
    pred_len = equipment_segment_lengths(pred_obj).reshape(-1, 4)
    return sum(pred_len[:, i].var(unbiased=False) for i in range(4))


def symmetry_loss(pred_obj: torch.Tensor) -> torch.Tensor:
    """Encourage left/right symmetry by matching mean segment lengths."""
    pred_len = equipment_segment_lengths(pred_obj).reshape(-1, 4)
    return torch.abs(pred_len[:, 0].mean() - pred_len[:, 1].mean()) + torch.abs(
        pred_len[:, 2].mean() - pred_len[:, 3].mean()
    )


def absolute_length_loss(pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
    """Supervise absolute equipment segment lengths with SmoothL1."""
    pred_len = equipment_segment_lengths(pred_obj)
    gt_len = equipment_segment_lengths(gt_obj)
    return torch.nn.functional.smooth_l1_loss(pred_len, gt_len)



def temporal_smoothness_loss(obj: torch.Tensor) -> torch.Tensor:
    """Penalize frame-to-frame equipment motion jitter.

    Expects sequence predictions shaped [B,T,4,2,3] or [B,T,8,3].
    Single-frame sequences return zero on the correct device/dtype.
    """
    obj = to_equip_point_repr(obj)
    if obj.ndim < 5:
        return obj.new_zeros(())
    if obj.shape[-4] <= 1:
        return obj.new_zeros(())
    velocity = obj[..., 1:, :, :, :] - obj[..., :-1, :, :, :]
    return torch.norm(velocity, dim=-1).mean()
