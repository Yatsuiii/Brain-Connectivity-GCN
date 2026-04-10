from __future__ import annotations

import torch
from torch import nn

from brain_gcn.utils.graph_conv import calculate_laplacian_with_self_loop


class GraphLinear(nn.Module):
    """Apply normalized adjacency, then a learned linear projection."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(adj_norm, x)
        return self.linear(x)


class GraphTemporalEncoder(nn.Module):
    """Graph-aware temporal encoder for ROI-level window sequences."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_graph = GraphLinear(1, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bold_windows: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # bold_windows: (B, W, N) -> node-major temporal sequence: (B*N, W, H)
        batch_size, num_windows, num_nodes = bold_windows.shape
        graph_steps = []
        for step in range(num_windows):
            x_t = bold_windows[:, step, :].unsqueeze(-1)
            step_adj = adj_norm[:, step, :, :] if adj_norm.dim() == 4 else adj_norm
            graph_steps.append(self.input_graph(x_t, step_adj))

        x = torch.stack(graph_steps, dim=1)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, num_windows, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :].reshape(batch_size, num_nodes, -1)
        return self.dropout(self.norm(x))


class AttentionReadout(nn.Module):
    """Learn ROI weights for subject-level graph pooling."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(node_embeddings).squeeze(-1), dim=-1)
        pooled = torch.sum(node_embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


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


class BrainGCNClassifier(nn.Module):
    """Subject-level ASD/TD classifier for dynamic brain connectivity."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        readout: str = "attention",
    ):
        super().__init__()
        if readout not in {"mean", "attention"}:
            raise ValueError("readout must be 'mean' or 'attention'")

        self.encoder = GraphTemporalEncoder(hidden_dim=hidden_dim, dropout=min(dropout, 0.2))
        self.readout = readout
        self.attention = AttentionReadout(hidden_dim) if readout == "attention" else None
        self.head = make_classifier_head(hidden_dim, num_classes, dropout)

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        adj_norm = calculate_laplacian_with_self_loop(adj)
        node_embeddings = self.encoder(bold_windows, adj_norm)
        pooled, attention_weights = graph_readout(node_embeddings, self.attention)
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights
        return logits


class GraphOnlyClassifier(nn.Module):
    """GCN baseline using each ROI's average window signal as node input."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        readout: str = "attention",
    ):
        super().__init__()
        if readout not in {"mean", "attention"}:
            raise ValueError("readout must be 'mean' or 'attention'")

        self.graph1 = GraphLinear(1, hidden_dim)
        self.graph2 = GraphLinear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionReadout(hidden_dim) if readout == "attention" else None
        self.head = make_classifier_head(hidden_dim, num_classes, dropout)

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        adj_norm = calculate_laplacian_with_self_loop(adj)
        if adj_norm.dim() == 4:
            adj_norm = adj_norm.mean(dim=1)
        x = bold_windows.mean(dim=1).unsqueeze(-1)
        x = torch.relu(self.graph1(x, adj_norm))
        x = self.dropout(torch.relu(self.graph2(x, adj_norm)))
        node_embeddings = self.norm(x)
        pooled, attention_weights = graph_readout(node_embeddings, self.attention)
        logits = self.head(pooled)
        if return_attention:
            return logits, attention_weights
        return logits


class TemporalGRUClassifier(nn.Module):
    """Temporal baseline over ROI vectors, with no graph message passing."""

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
    """Static FC baseline over the upper triangle of the adjacency matrix."""

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


def build_model(
    model_name: str,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.5,
    readout: str = "attention",
) -> nn.Module:
    if model_name == "graph_temporal":
        return BrainGCNClassifier(hidden_dim, num_classes, dropout, readout)
    if model_name == "gcn":
        return GraphOnlyClassifier(hidden_dim, num_classes, dropout, readout)
    if model_name == "gru":
        return TemporalGRUClassifier(hidden_dim, num_classes, dropout)
    if model_name == "fc_mlp":
        return ConnectivityMLPClassifier(hidden_dim, num_classes, dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
