"""
Dynamic FC Temporal Attention model for ASD/TD classification.

Architecture (STAGIN-inspired, simplified):
  Input  : (B, W, N) — per-window ROI connectivity strength (mean |FC| per ROI)
  Step 1 : Linear projection N → H
  Step 2 : Learnable positional encoding over W time steps
  Step 3 : Transformer encoder (multi-head self-attention over windows)
  Step 4 : Attention-weighted pooling over W → subject embedding (H,)
  Step 5 : MLP classifier → 2

Why this works:
  ASD shows altered *dynamic* connectivity — not just different mean FC but
  different temporal patterns of connectivity fluctuation across brain states.
  The self-attention learns which window combinations are most discriminative.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DynamicFCAttention(nn.Module):

    def __init__(
        self,
        num_rois: int = 200,
        max_windows: int = 30,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Project ROI connectivity strengths to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_rois, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Learnable positional encoding — one vector per window
        self.pos_embed = nn.Parameter(torch.randn(1, max_windows, hidden_dim) * 0.02)

        # Transformer encoder: self-attention over time windows
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout * 0.5,
            batch_first=True,
            norm_first=True,             # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling over time: learn which windows matter
        self.time_attn = nn.Linear(hidden_dim, 1)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor | None = None,   # unused — kept for interface compatibility
        return_attention: bool = False,
    ) -> torch.Tensor:
        # bold_windows: (B, W, N) — mean |FC| per ROI per time window
        B, W, N = bold_windows.shape

        # Project each window's ROI features to hidden dim
        x = self.input_proj(bold_windows)             # (B, W, H)

        # Add positional encoding
        x = x + self.pos_embed[:, :W, :]

        # Self-attention over time windows
        x = self.transformer(x)                        # (B, W, H)

        # Attention-weighted pooling: which windows are most discriminative?
        attn = torch.softmax(self.time_attn(x).squeeze(-1), dim=1)   # (B, W)
        embedding = (x * attn.unsqueeze(-1)).sum(dim=1)               # (B, H)

        logits = self.head(embedding)

        if return_attention:
            return logits, attn
        return logits
