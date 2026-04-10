"""
Brain GCN model definitions.

v2 changes:
  - TwoLayerGCN with residual connection replaces single GraphLinear in encoder
  - DropEdge applied in BrainGCNClassifier.forward() during training
  - GraphOnlyClassifier also upgraded to TwoLayerGCN (was already 2-layer but
    without residual or LayerNorm between layers)
"""

from __future__ import annotations

import torch
from torch import nn

from brain_gcn.utils.graph_conv import calculate_laplacian_with_self_loop, drop_edge


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GraphLinear(nn.Module):
    """Apply normalized adjacency, then a learned linear projection."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(adj_norm, x)
        return self.linear(x)


class TwoLayerGCN(nn.Module):
    """2-layer GCN with residual skip connection.

    Architecture (Kipf & Welling 2017 + He et al. 2016 residuals):
        h1  = ReLU(LayerNorm(GCN1(x)))
        h2  = Dropout(ReLU(LayerNorm(GCN2(h1))))
        out = h2 + skip(x)   # skip is a plain linear projection

    The residual stabilises gradient flow and lets the model interpolate
    between 1-hop and 2-hop aggregation.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GraphLinear(in_dim, hidden_dim)
        self.gcn2 = GraphLinear(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.gcn1(x, adj_norm)))
        h = self.drop(torch.relu(self.norm2(self.gcn2(h, adj_norm))))
        return h + self.skip(x)          # residual


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class GraphTemporalEncoder(nn.Module):
    """Graph-aware temporal encoder for ROI-level window sequences.

    Vectorized implementation:
      Reshape bold_windows to (B*W, N, 1) for a single batched GCN pass.
      Then reshape back and apply node-major GRU.
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_graph = TwoLayerGCN(1, hidden_dim, dropout=min(dropout, 0.1))
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bold_windows: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # bold_windows: (B, W, N)
        batch_size, num_windows, num_nodes = bold_windows.shape

        # Vectorized: reshape to (B*W, N, 1) for single GCN pass
        x = bold_windows.reshape(batch_size * num_windows, num_nodes, 1)  # (B*W, N, 1)

        # Handle both 3D (B,N,N) and 4D (B,W,N,N) adjacency
        if adj_norm.dim() == 4:
            # (B, W, N, N) → (B*W, N, N)
            adj_flat = adj_norm.reshape(batch_size * num_windows, num_nodes, num_nodes)
        else:
            # (B, N, N) → replicate for all windows
            adj_flat = adj_norm.unsqueeze(1).expand(-1, num_windows, -1, -1)
            adj_flat = adj_flat.reshape(batch_size * num_windows, num_nodes, num_nodes)

        # Single batched GCN pass → (B*W, N, H)
        h = self.input_graph(x, adj_flat)

        # Reshape back and apply node-major GRU
        h = h.reshape(batch_size, num_windows, num_nodes, -1)  # (B, W, N, H)
        hidden_dim = h.shape[-1]
        h = h.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, num_windows, hidden_dim)
        h, _ = self.gru(h)
        h = h[:, -1, :].reshape(batch_size, num_nodes, -1)  # (B, N, H)
        return self.dropout(self.norm(h))


class AttentionReadout(nn.Module):
    """Learn per-ROI attention weights for subject-level graph pooling.
    
    Single linear projection is sufficient for N=200 nodes.
    More interpretable and faster than 2-layer MLP.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(node_embeddings).squeeze(-1), dim=-1)
        pooled = torch.sum(node_embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


# ---------------------------------------------------------------------------
# Helpers shared across classifiers
# ---------------------------------------------------------------------------

def make_classifier_head(hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def graph_readout(
    node_embeddings: torch.Tensor,
    attention: AttentionReadout | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if attention is None:
        return node_embeddings.mean(dim=1), None
    return attention(node_embeddings)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class BrainGCNClassifier(nn.Module):
    """Subject-level ASD/TD classifier for dynamic brain connectivity.

    v2: TwoLayerGCN encoder + DropEdge during training.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        readout: str = "attention",
        drop_edge_p: float = 0.1,
    ):
        super().__init__()
        if readout not in {"mean", "attention"}:
            raise ValueError("readout must be 'mean' or 'attention'")

        self.encoder = GraphTemporalEncoder(hidden_dim=hidden_dim, dropout=min(dropout, 0.2))
        self.readout = readout
        self.attention = AttentionReadout(hidden_dim) if readout == "attention" else None
        self.head = make_classifier_head(hidden_dim, num_classes, dropout)
        self.drop_edge_p = drop_edge_p

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        # DropEdge: applied before Laplacian normalisation, training only
        adj = drop_edge(adj, p=self.drop_edge_p, training=self.training)
        adj_norm = calculate_laplacian_with_self_loop(adj)
        node_embeddings = self.encoder(bold_windows, adj_norm)
        pooled, attention_weights = graph_readout(node_embeddings, self.attention)
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights
        return logits


class GraphOnlyClassifier(nn.Module):
    """GCN baseline — each ROI's average window signal as node input.

    v2: upgraded to TwoLayerGCN with residual + DropEdge.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        readout: str = "attention",
        drop_edge_p: float = 0.1,
    ):
        super().__init__()
        if readout not in {"mean", "attention"}:
            raise ValueError("readout must be 'mean' or 'attention'")

        self.gcn = TwoLayerGCN(1, hidden_dim, dropout=min(dropout, 0.1))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionReadout(hidden_dim) if readout == "attention" else None
        self.head = make_classifier_head(hidden_dim, num_classes, dropout)
        self.drop_edge_p = drop_edge_p

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        adj = drop_edge(adj, p=self.drop_edge_p, training=self.training)
        adj_norm = calculate_laplacian_with_self_loop(adj)
        if adj_norm.dim() == 4:
            adj_norm = adj_norm.mean(dim=1)
        x = bold_windows.mean(dim=1).unsqueeze(-1)     # (B, N, 1)
        x = self.dropout(self.norm(self.gcn(x, adj_norm)))
        pooled, attention_weights = graph_readout(x, self.attention)
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights
        return logits


class TemporalGRUClassifier(nn.Module):
    """Temporal baseline — GRU over ROI vectors, no graph message passing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj = nn.LazyLinear(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = make_classifier_head(hidden_dim, num_classes, dropout)

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        x = torch.relu(self.input_proj(bold_windows))
        x, _ = self.gru(x)
        x = self.dropout(self.norm(x[:, -1, :]))
        logits = self.head(x)
        if return_attention:
            return logits, None
        return logits


class ConnectivityMLPClassifier(nn.Module):
    """Static FC baseline — upper triangle of adjacency matrix as features."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if adj.dim() == 4:
            adj = adj.mean(dim=1)
        row, col = torch.triu_indices(adj.size(-2), adj.size(-1), offset=1, device=adj.device)
        x = adj[:, row, col]
        logits = self.net(x)
        if return_attention:
            return logits, None
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.5,
    readout: str = "attention",
    drop_edge_p: float = 0.1,
) -> nn.Module:
    if model_name == "graph_temporal":
        return BrainGCNClassifier(hidden_dim, num_classes, dropout, readout, drop_edge_p)
    if model_name == "gcn":
        return GraphOnlyClassifier(hidden_dim, num_classes, dropout, readout, drop_edge_p)
    if model_name == "gru":
        return TemporalGRUClassifier(hidden_dim, num_classes, dropout)
    if model_name == "fc_mlp":
        return ConnectivityMLPClassifier(hidden_dim, num_classes, dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
