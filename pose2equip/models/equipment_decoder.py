"""Transformer decoders for equipment endpoint prediction."""

import torch
import torch.nn as nn


class EquipmentQueryDecoder(nn.Module):
    def __init__(
        self,
        num_queries=4,
        dim=256,
        num_heads=8,
        num_layers=3,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(temporal_layer, num_layers=2)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),
        )

    def forward(self, memory):
        """Decode memory [B,T,N,C] into equipment endpoints [B,T,Q,2,3]."""
        b, t, n, c = memory.shape
        memory = memory.reshape(b * t, n, c)
        queries = self.queries.unsqueeze(0).repeat(b * t, 1, 1)
        z = self.decoder(tgt=queries, memory=memory)
        z = z.reshape(b, t, self.num_queries, c)

        q = z.shape[2]
        z = z.permute(0, 2, 1, 3).reshape(b * q, t, c)
        z = self.temporal_attn(z)
        z = z.reshape(b, q, t, c).permute(0, 2, 1, 3)

        pred = self.reg_head(z)
        return pred.reshape(b, t, q, 2, 3)
