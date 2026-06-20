"""Pose-only backbones for human skeleton to equipment keypoint prediction."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from pose2equip.models.equipment_decoder import EquipmentQueryDecoder
from pose2equip.models.pose_encoder import PoseEncoder


def _normalize_human_3d(human_3d: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if human_3d.ndim == 3:
        return human_3d.unsqueeze(1), True
    if human_3d.ndim == 4:
        return human_3d, False
    raise ValueError(
        f"Expected human_3d shape [B,J,3] or [B,T,J,3], got {tuple(human_3d.shape)}"
    )


def _restore_output_shape(pred_obj: torch.Tensor, single_frame_mode: bool) -> torch.Tensor:
    if single_frame_mode:
        return pred_obj.squeeze(1)
    return pred_obj


def _validate_even_equipment_points(num_equip_kpts: int) -> int:
    num_equip_kpts = int(num_equip_kpts)
    if num_equip_kpts % 2 != 0:
        raise ValueError(
            f"num_equip_kpts must be even (pairs of endpoints), got {num_equip_kpts}"
        )
    return num_equip_kpts // 2


def _sinusoidal_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class MLPBaselineNet(nn.Module):
    """Per-frame MLP baseline: flattened skeleton -> equipment endpoints."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.num_equip = _validate_even_equipment_points(num_equip_kpts)
        in_dim = self.num_joints * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_equip * 2 * 3),
        )

    def forward(self, human_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        human_3d, single_frame_mode = _normalize_human_3d(human_3d)
        b, t, j, c = human_3d.shape
        if j != self.num_joints or c != 3:
            raise ValueError(f"Expected [B,T,{self.num_joints},3], got {tuple(human_3d.shape)}")
        pred = self.net(human_3d.reshape(b * t, j * c))
        pred = pred.reshape(b, t, self.num_equip, 2, 3)
        return {"object_3d": _restore_output_shape(pred, single_frame_mode)}


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GroupNorm(1, hidden_dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TCNBaselineNet(nn.Module):
    """Temporal convolution baseline over flattened skeleton features."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.num_equip = _validate_even_equipment_points(num_equip_kpts)
        in_dim = self.num_joints * 3
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[TemporalConvBlock(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, self.num_equip * 2 * 3)

    def forward(self, human_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        human_3d, single_frame_mode = _normalize_human_3d(human_3d)
        b, t, j, c = human_3d.shape
        if j != self.num_joints or c != 3:
            raise ValueError(f"Expected [B,T,{self.num_joints},3], got {tuple(human_3d.shape)}")
        x = self.input_proj(human_3d.reshape(b, t, j * c))
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        pred = self.head(x).reshape(b, t, self.num_equip, 2, 3)
        return {"object_3d": _restore_output_shape(pred, single_frame_mode)}


class SkeletonTransformerNet(nn.Module):
    """Spatio-temporal Transformer over skeleton joint tokens."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.hidden_dim = int(hidden_dim)
        self.num_equip = _validate_even_equipment_points(num_equip_kpts)
        self.coord_proj = nn.Linear(3, hidden_dim)
        self.joint_embed = nn.Parameter(torch.zeros(self.num_joints, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_equip * 2 * 3),
        )
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

    def forward(self, human_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        human_3d, single_frame_mode = _normalize_human_3d(human_3d)
        b, t, j, c = human_3d.shape
        if j != self.num_joints or c != 3:
            raise ValueError(f"Expected [B,T,{self.num_joints},3], got {tuple(human_3d.shape)}")
        x = self.coord_proj(human_3d)
        x = x + self.joint_embed.view(1, 1, j, self.hidden_dim)
        time_pe = _sinusoidal_encoding(t, self.hidden_dim, human_3d.device)
        x = x + time_pe.view(1, t, 1, self.hidden_dim)
        x = x.reshape(b, t * j, self.hidden_dim)
        x = self.encoder(x).reshape(b, t, j, self.hidden_dim)
        frame_feat = x.mean(dim=2)
        pred = self.head(frame_feat).reshape(b, t, self.num_equip, 2, 3)
        return {"object_3d": _restore_output_shape(pred, single_frame_mode)}


class STGCNQueryNet(nn.Module):
    """ST-GCN pose encoder followed by equipment query decoder."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        target_skeleton_connections_idx: Optional[list[tuple[int, int]]] = None,
        decoder_layers: int = 3,
        num_heads: int = 8,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_equip = _validate_even_equipment_points(num_equip_kpts)
        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )
        self.decoder = EquipmentQueryDecoder(
            num_queries=self.num_equip,
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=decoder_layers,
        )

    def forward(self, human_3d: torch.Tensor) -> dict[str, torch.Tensor]:
        human_3d, single_frame_mode = _normalize_human_3d(human_3d)
        memory = self.pose_encoder(human_3d)
        pred = self.decoder(memory)
        return {"object_3d": _restore_output_shape(pred, single_frame_mode)}


def build_skeleton_backbone(
    name: str,
    num_joints: int = 15,
    hidden_dim: int = 256,
    num_equip_kpts: int = 8,
    target_skeleton_connections_idx: Optional[list[tuple[int, int]]] = None,
    num_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.1,
    kernel_size: int = 5,
    decoder_layers: Optional[int] = None,
) -> nn.Module:
    normalized = name.lower()
    common = {
        "num_joints": num_joints,
        "hidden_dim": hidden_dim,
        "num_equip_kpts": num_equip_kpts,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "kernel_size": kernel_size,
    }
    if normalized == "mlp":
        return MLPBaselineNet(**common)
    if normalized == "tcn":
        return TCNBaselineNet(**common)
    if normalized in {"skeleton_transformer", "transformer"}:
        return SkeletonTransformerNet(**common)
    if normalized in {"stgcn_query", "st-gcn-query"}:
        return STGCNQueryNet(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            num_equip_kpts=num_equip_kpts,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
            decoder_layers=decoder_layers if decoder_layers is not None else num_layers,
            num_heads=num_heads,
        )
    raise ValueError(
        f"Unsupported skeleton backbone '{name}'. Expected one of: mlp, tcn, skeleton_transformer, stgcn_query."
    )
