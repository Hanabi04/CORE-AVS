"""Class-indexed dual-risk auditor used by the released checkpoints."""

import torch
from torch import nn


class SetAuditor(nn.Module):
    """Encode 21 class tokens and predict mask risk and three audio states."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(13, 32)
        self.class_embedding = nn.Embedding(21, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(32)
        self.risk = nn.Linear(32, 1)
        self.corruption = nn.Linear(32, 3)

    def forward(self, value):
        """Map [batch, 21, 13] tokens to [batch] risk and [batch, 3] logits."""
        if value.ndim != 3 or value.shape[1:] != (21, 13):
            raise ValueError("Expected batch x 21 classes x 13 features")
        classes = torch.arange(21, device=value.device)
        encoded = self.input(value) + self.class_embedding(classes)[None]
        pooled = self.norm(self.encoder(encoded)).mean(1)
        return torch.sigmoid(self.risk(pooled)[:, 0]), self.corruption(pooled)
