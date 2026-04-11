"""
Advanced model architectures for brain connectivity analysis.

New models:
- Graph Attention Networks (GAT)
- Transformer-based temporal encoder
- 3D-CNN for spatiotemporal features
- GraphSAGE (sampling-aggregating)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from brain_gcn.utils.graph_conv import calculate_laplacian_with_self_loop, drop_edge


# ---------------------------------------------------------------------------
# Graph Attention Networks (GAT)
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, nodes, in_dim)
        # adj: (batch, nodes, nodes) or (nodes, nodes)
        
        Q = self.query(x)  # (batch, nodes, out_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head: (batch, nodes, heads, head_dim)
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, heads, nodes, nodes)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores + (1 - adj.unsqueeze(1)) * -1e9  # Mask non-edges

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, heads, nodes, head_dim)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)  # (batch, nodes, out_dim)

        return self.fc_out(out)


class GATEncoder(nn.Module):
    """Multi-layer Graph Attention Network."""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layer1 = GraphAttentionLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.layer2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.dropout(F.relu(self.norm1(h)))
        h = self.layer2(h, adj)
        h = self.dropout(F.relu(self.norm2(h)))
        return h


# ---------------------------------------------------------------------------
# Transformer-based Temporal Encoder
# ---------------------------------------------------------------------------

class TransformerTemporalEncoder(nn.Module):
    """Transformer-based encoder for temporal sequences."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bold_windows: torch.Tensor) -> torch.Tensor:
        # bold_windows: (batch, windows, nodes) → embed → (batch * nodes, windows, hidden_dim)
        batch, windows, nodes = bold_windows.shape
        
        # Embed time dimension
        x = bold_windows.permute(0, 2, 1).reshape(batch * nodes, windows, 1)  # (B*N, W, 1)
        x = self.embedding(x)  # (B*N, W, hidden_dim)
        
        # Transformer
        h = self.transformer(x)  # (B*N, W, hidden_dim)
        h = self.norm(h)
        h = h[:, -1, :]  # Take last token
        h = h.reshape(batch, nodes, -1)  # (B, N, hidden_dim)
        
        return h


# ---------------------------------------------------------------------------
# 3D-CNN for Spatiotemporal Features
# ---------------------------------------------------------------------------

class CNN3D(nn.Module):
    """3D-CNN for spatiotemporal brain connectivity analysis."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        # Input: (batch, 1, time, height, width) for connectivity matrices
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(dropout)
        self.norm1 = nn.BatchNorm3d(16)
        self.norm2 = nn.BatchNorm3d(32)
        self.norm3 = nn.BatchNorm3d(64)

    def forward(self, fc_windows: torch.Tensor) -> torch.Tensor:
        # fc_windows: (batch, windows, nodes, nodes)
        batch, windows, nodes, _ = fc_windows.shape
        
        # Add channel dimension: (batch, 1, windows, nodes, nodes)
        x = fc_windows.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = x.mean(dim=(2, 3, 4))  # (batch, channels)
        return x


# ---------------------------------------------------------------------------
# GraphSAGE (Sampling and Aggregating)
# ---------------------------------------------------------------------------

class GraphSAGELayer(nn.Module):
    """GraphSAGE layer using mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.agg_weight = nn.Linear(in_dim, out_dim)
        self.self_weight = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (batch, nodes, in_dim)
        # adj: (batch, nodes, nodes) or (nodes, nodes)
        
        # Aggregate neighbors: (batch, nodes, in_dim)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        
        # Normalize adjacency for aggregation
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree
        
        neighbor_agg = torch.bmm(adj_norm, x)  # (batch, nodes, in_dim)
        
        # Combine self and aggregated neighbor features
        h_agg = self.agg_weight(neighbor_agg)
        h_self = self.self_weight(x)
        h = h_agg + h_self
        h = F.relu(self.norm(h))
        h = self.dropout(h)
        
        return h


class GraphSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, knockout: float = 0.1):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim, dropout=knockout)
        self.layer2 = GraphSAGELayer(hidden_dim, hidden_dim, dropout=knockout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h


# ---------------------------------------------------------------------------
# Attention Readout (shared)
# ---------------------------------------------------------------------------

class AttentionReadout(nn.Module):
    """Learn per-ROI attention weights."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(node_embeddings).squeeze(-1), dim=-1)
        pooled = torch.sum(node_embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


# ---------------------------------------------------------------------------
# Classifier Heads
# ---------------------------------------------------------------------------

def make_head(hidden_dim: int, num_classes: int = 2, dropout: float = 0.5) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


# ---------------------------------------------------------------------------
# Complete Models
# ---------------------------------------------------------------------------

class GATClassifier(nn.Module):
    """Graph Attention Network classifier."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.encoder = GATEncoder(1, hidden_dim, num_heads=num_heads, dropout=min(dropout, 0.2))
        self.attention = AttentionReadout(hidden_dim)
        self.head = make_head(hidden_dim, dropout=dropout)

    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, windows, nodes = bold_windows.shape
        
        # Process each window
        embeddings_list = []
        adj_norm = calculate_laplacian_with_self_loop(adj)
        
        for w in range(windows):
            x = bold_windows[:, w, :].unsqueeze(-1)  # (batch, nodes, 1)
            if adj_norm.dim() == 3:
                adj_w = adj_norm
            else:
                adj_w = adj_norm.unsqueeze(0)
            h = self.encoder(x, adj_w)
            embeddings_list.append(h)
        
        # Average over windows
        h = torch.stack(embeddings_list, dim=1).mean(dim=1)  # (batch, nodes, hidden_dim)
        
        pooled, _ = self.attention(h)
        logits = self.head(pooled)
        return logits


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for temporal brain signals."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.temporal_encoder = TransformerTemporalEncoder(hidden_dim, num_heads=num_heads, dropout=min(dropout, 0.2))
        self.attention = AttentionReadout(hidden_dim)
        self.head = make_head(hidden_dim, dropout=dropout)

    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.temporal_encoder(bold_windows)  # (batch, nodes, hidden_dim)
        pooled, _ = self.attention(h)
        logits = self.head(pooled)
        return logits


class CNN3DClassifier(nn.Module):
    """3D-CNN classifier for connectivity dynamics."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.cnn = CNN3D(hidden_dim, dropout=min(dropout, 0.2))
        self.head = make_head(64, dropout=dropout)

    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Reshape adj to (batch, 1, nodes, nodes) if needed for 3D convolution
        # Use adjacency as a static connectivity pattern expanded across windows
        batch_size, nodes, nodes2 = adj.shape if adj.dim() == 3 else (adj.shape[0], adj.shape[1], adj.shape[2])
        
        # Reshape for 3D CNN: (batch, 1, windows, nodes, nodes)
        # Use the adjacency pattern expanded across windows
        fc_windows = adj.unsqueeze(1).expand(-1, bold_windows.shape[1] if bold_windows.dim() > 2 else 1, -1, -1)
        
        h = self.cnn(fc_windows)  # (batch, 64)
        logits = self.head(h)
        return logits


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE-based classifier."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.encoder = GraphSAGEEncoder(1, hidden_dim, knockout=min(dropout, 0.2))
        self.attention = AttentionReadout(hidden_dim)
        self.head = make_head(hidden_dim, dropout=dropout)

    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, windows, nodes = bold_windows.shape
        
        adj_norm = calculate_laplacian_with_self_loop(adj)
        embeddings_list = []
        
        for w in range(windows):
            x = bold_windows[:, w, :].unsqueeze(-1)  # (batch, nodes, 1)
            if adj_norm.dim() == 3:
                adj_w = adj_norm
            else:
                adj_w = adj_norm.unsqueeze(0)
            h = self.encoder(x, adj_w)
            embeddings_list.append(h)
        
        h = torch.stack(embeddings_list, dim=1).mean(dim=1)
        pooled, _ = self.attention(h)
        logits = self.head(pooled)
        return logits
