"""Image encoders used by Pose2Equip models."""

import torch
import torch.nn as nn
from transformers import AutoModel


class DinoPatchEncoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov2-base",
        out_dim=256,
        freeze=True,
        mock=False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        self.encoder.requires_grad_(not freeze)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        """Encode image sequences from [B,T,3,H,W] to [B,T,N,C]."""
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if x.max() > 1:
            x = x / 255.0

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.encoder(pixel_values=x)
            feat = outputs.last_hidden_state[:, 1:]

        feat = self.proj(feat)
        n = feat.shape[1]
        return feat.reshape(b, t, n, -1)
