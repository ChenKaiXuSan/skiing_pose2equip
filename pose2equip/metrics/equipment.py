"""Metrics for 3D equipment keypoint prediction."""

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np


def to_flat_n83(obj: np.ndarray) -> np.ndarray:
    """Normalize object tensor to [N, 8, 3] for metric computation."""
    arr = np.asarray(obj)
    if tuple(arr.shape[-3:]) == (4, 2, 3):
        return arr.reshape(-1, 8, 3)
    if tuple(arr.shape[-2:]) == (8, 3):
        return arr.reshape(-1, 8, 3)
    raise ValueError(
        f"Expected object shape ending with [4,2,3] or [8,3], got {tuple(arr.shape)}"
    )


def compute_procrustes_alignment(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Find rotation, scale, and translation that align pred to gt."""
    pred_mean = pred.mean(axis=0, keepdims=True)
    gt_mean = gt.mean(axis=0, keepdims=True)
    pred_c = pred - pred_mean
    gt_c = gt - gt_mean

    h = pred_c.T @ gt_c
    u, _, vt = np.linalg.svd(h)
    r = (u @ vt).astype(np.float32)
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = (u @ vt).astype(np.float32)

    scale = float(np.linalg.norm(gt_c) / (np.linalg.norm(pred_c) + 1e-8))
    pred_aligned = (scale * pred_c @ r.T) + gt_mean
    alignment_error = float(np.linalg.norm(pred_aligned - gt, axis=1).mean())
    return pred_aligned, scale, alignment_error


def evaluate_pose_metrics(pred_obj: np.ndarray, gt_obj: np.ndarray) -> Dict[str, float]:
    """Evaluate global and per-equipment MPJPE metrics."""
    pred_obj = to_flat_n83(pred_obj)
    gt_obj = to_flat_n83(gt_obj)

    mpjpe_val = np.linalg.norm(pred_obj - gt_obj, axis=2).mean()
    pred_flat = pred_obj.reshape(-1, 3)
    gt_flat = gt_obj.reshape(-1, 3)
    pred_aligned, _, _ = compute_procrustes_alignment(pred_flat, gt_flat)
    pa_mpjpe = np.linalg.norm(pred_aligned - gt_flat, axis=1).mean()

    left_ski_err = np.linalg.norm(pred_obj[:, :2] - gt_obj[:, :2], axis=2).mean()
    right_ski_err = np.linalg.norm(pred_obj[:, 2:4] - gt_obj[:, 2:4], axis=2).mean()
    left_pole_err = np.linalg.norm(pred_obj[:, 4:6] - gt_obj[:, 4:6], axis=2).mean()
    right_pole_err = np.linalg.norm(pred_obj[:, 6:8] - gt_obj[:, 6:8], axis=2).mean()

    return {
        "mpjpe": float(mpjpe_val),
        "pa_mpjpe": float(pa_mpjpe),
        "mpjpe_left_ski": float(left_ski_err),
        "mpjpe_right_ski": float(right_ski_err),
        "mpjpe_left_pole": float(left_pole_err),
        "mpjpe_right_pole": float(right_pole_err),
    }



def save_evaluation_report(
    metrics: Dict[str, float],
    total_samples: int,
    output_dir: str | Path,
    stem: str = "evaluation_metrics",
) -> Dict[str, Path]:
    """Save evaluation metrics as both human-readable text and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ski_avg = (metrics["mpjpe_left_ski"] + metrics["mpjpe_right_ski"]) / 2.0
    pole_avg = (metrics["mpjpe_left_pole"] + metrics["mpjpe_right_pole"]) / 2.0
    payload = {
        "total_samples": int(total_samples),
        **{k: float(v) for k, v in metrics.items()},
        "mpjpe_avg_ski": float(ski_avg),
        "mpjpe_avg_pole": float(pole_avg),
    }

    txt_path = output_dir / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Equipment 3D Keypoint Prediction - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples evaluated: {total_samples}\n\n")
        f.write("Global Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  MPJPE (Mean Per Joint Position Error):  {metrics['mpjpe']:.4f} mm\n")
        f.write(f"  PA-MPJPE (Procrustes Aligned):         {metrics['pa_mpjpe']:.4f} mm\n\n")
        f.write("Per-Object Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Left Ski MPJPE:   {metrics['mpjpe_left_ski']:.4f} mm\n")
        f.write(f"  Right Ski MPJPE:  {metrics['mpjpe_right_ski']:.4f} mm\n")
        f.write(f"  Left Pole MPJPE:  {metrics['mpjpe_left_pole']:.4f} mm\n")
        f.write(f"  Right Pole MPJPE: {metrics['mpjpe_right_pole']:.4f} mm\n\n")
        f.write(f"  Avg Ski Error:    {ski_avg:.4f} mm\n")
        f.write(f"  Avg Pole Error:   {pole_avg:.4f} mm\n\n")
        f.write("=" * 60 + "\n")

    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return {"txt": txt_path, "json": json_path}
