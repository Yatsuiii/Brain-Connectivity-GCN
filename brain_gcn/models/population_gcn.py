"""
Population-level GCN for subject-level ASD/TD classification.

All subjects are nodes in a single graph — transductive setting.
The model sees all node features (including unlabeled val/test subjects)
during forward passes; loss is masked to training nodes only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GraphConv(nn.Module):
    """Single graph convolution: linear projection after neighborhood aggregation."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: pre-normalized (N, N); x: (N, in_dim)
        return self.linear(adj @ x)


class PopulationGCN(nn.Module):
    """2-layer GCN on the subject population graph.

    Architecture
    ============
    Input → Dropout → GC1 → LayerNorm → ReLU
          → Dropout → GC2 → LayerNorm → ReLU
          → Dropout → Linear → logits (N, num_classes)

    Depth 2 is sufficient: each node aggregates 2-hop neighbors,
    covering subjects with similar age+sex across the whole cohort.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gc1   = GraphConv(in_dim, hidden_dim)
        self.gc2   = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head  = nn.Linear(hidden_dim, num_classes)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = F.relu(self.norm1(self.gc1(x, adj)))
        x = self.drop(x)
        x = F.relu(self.norm2(self.gc2(x, adj)))
        x = self.drop(x)
        return self.head(x)                           # (N, num_classes)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return post-GC2 embeddings for t-SNE / analysis."""
        x = self.gc1(x, adj)
        x = F.relu(self.norm1(x))
        x = self.gc2(x, adj)
        return F.relu(self.norm2(x))
