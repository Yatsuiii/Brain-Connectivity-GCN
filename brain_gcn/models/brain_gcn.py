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
from brain_gcn.utils.grl import GradientReversal


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

    @staticmethod
    def _fc_features(adj: torch.Tensor) -> torch.Tensor:
        """Extract features from adj tensor (various shapes):

        (B, N, N)    → (B, N*(N-1)/2)   signed mean FC upper triangle
        (B, 2, N, N) → (B, N*(N-1))     mean FC || std FC concatenated
        (B, 1, K)    → (B, K)           pre-computed PCA features (pass-through)
        (B, W, N, N) → (B, N*(N-1)/2)   dynamic seq: averaged over windows first
        """
        if adj.dim() == 3:
            if adj.size(1) == 1:
                # PCA projection already computed in dataset — just flatten
                return adj.squeeze(1)                        # (B, K)
            # (B, N, N) — standard case
            row, col = torch.triu_indices(adj.size(-2), adj.size(-1), offset=1,
                                          device=adj.device)
            return adj[:, row, col]                          # (B, 19900)

        if adj.dim() == 4:
            if adj.size(1) == 2:
                # [mean_fc, std_fc] channels
                row, col = torch.triu_indices(adj.size(-2), adj.size(-1), offset=1,
                                              device=adj.device)
                x_mean = adj[:, 0, row, col]
                x_std  = adj[:, 1, row, col]
                return torch.cat([x_mean, x_std], dim=-1)   # (B, 2*19900)
            # Dynamic window sequence: average then extract
            adj = adj.mean(dim=1)                            # (B, N, N)
            row, col = torch.triu_indices(adj.size(-2), adj.size(-1), offset=1,
                                          device=adj.device)
            return adj[:, row, col]

        raise ValueError(f"Unexpected adj shape: {tuple(adj.shape)}")

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        x = self._fc_features(adj)
        logits = self.net(x)
        if return_attention:
            return logits, None
        return logits


class BrainModeNetwork(nn.Module):
    """
    Novel architecture: Brain Mode Network (BMN).

    Learns K 'brain modes' — directions in ROI space (v_k ∈ R^N).
    Projects the N×N FC matrix into a compact K×K 'mode interaction matrix':

        M_kl = v_k^T · FC · v_l

    Diagonal M_kk measures connectivity energy along mode k (Rayleigh quotient).
    Off-diagonal M_kl captures cross-mode coupling between networks.

    With K=16 modes and N=200 ROIs: 136 features instead of 19,900.
    Inductive bias: each mode can specialize to a brain network community
    (e.g. DMN, FPN, SMN) — the model learns which communities matter for ASD.

    Orthogonality regularization keeps modes diverse (callable via
    orthogonality_loss(), weight controlled externally in the training task).
    """

    def __init__(
        self,
        num_nodes: int,
        num_modes: int = 16,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
        mode_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_nodes = num_nodes

        # Learnable modes: K × N — default initialization is near-orthonormal via QR.
        # Caller may pass a (K, N) tensor from discriminative_init() instead.
        if mode_init is not None:
            modes_init = mode_init.clone().float()
        else:
            modes_init_np = torch.randn(num_nodes, num_modes)
            Q, _ = torch.linalg.qr(modes_init_np)      # (N, K) orthonormal columns
            modes_init = Q.T.contiguous()               # (K, N)
        self.modes = nn.Parameter(modes_init)

        # Features: K(K+1)/2 from static M  +  K from temporal std(A_k)
        num_fc_features = num_modes * (num_modes + 1) // 2
        num_total_features = num_fc_features + num_modes   # static + dynamic

        self.classifier = nn.Sequential(
            nn.LayerNorm(num_total_features),
            nn.Linear(num_total_features, hidden_dim),
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
        # adj: (B, N, N) signed FC matrix; also accept (B, W, N, N) → avg over W
        if adj.dim() == 4:
            adj = adj.mean(dim=1)                            # (B, N, N)

        # ── Static stream: mode interaction matrix ──────────────────────────
        # M_kl = v_k^T · FC · v_l  →  (B, K, K)
        M = torch.einsum('kn,bnm,lm->bkl', self.modes, adj, self.modes)

        # Extract upper triangle (including diagonal): K(K+1)/2 features
        r, c = torch.triu_indices(self.num_modes, self.num_modes,
                                  offset=0, device=adj.device)
        fc_features = M[:, r, c]                            # (B, K(K+1)/2)

        # ── Dynamic stream: temporal variability of mode activity ───────────
        # A_k(t) = v_k · bold(t)  →  A: (B, W, K)
        # std(A_k) captures how much each network fluctuates over time.
        # This is genuinely new information not present in static mean FC.
        A = torch.einsum('kn,bwn->bwk', self.modes, bold_windows)  # (B, W, K)
        dyn_features = A.std(dim=1)                         # (B, K)

        features = torch.cat([fc_features, dyn_features], dim=-1)  # (B, K(K+1)/2+K)

        logits = self.classifier(features)
        if return_attention:
            return logits, None
        return logits

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalise non-orthonormal modes: ||V_norm @ V_norm^T - I||_F^2 / K^2.

        Encourages each mode to capture a distinct connectivity direction.
        Dividing by K^2 keeps the loss scale independent of num_modes.
        """
        V_norm = self.modes / (self.modes.norm(dim=1, keepdim=True) + 1e-8)
        gram = V_norm @ V_norm.T                            # (K, K)
        I = torch.eye(self.num_modes, device=gram.device, dtype=gram.dtype)
        return ((gram - I) ** 2).mean()

    @staticmethod
    def discriminative_init(
        train_fc_asd: "np.ndarray",
        train_fc_td: "np.ndarray",
        num_modes: int,
    ) -> "torch.Tensor":
        """Initialize modes from SVD of the ASD-TD mean FC difference matrix.

        The k-th left singular vector of (mean_FC_ASD − mean_FC_TD) is the k-th
        most discriminative direction in ROI space — the direction along which the
        two classes differ most. Starting here gives the optimizer a head start
        and reduces the number of epochs needed to learn discriminative modes.

        Parameters
        ----------
        train_fc_asd : (n_asd, N, N) FC matrices for ASD training subjects
        train_fc_td  : (n_td,  N, N) FC matrices for TD training subjects
        num_modes    : K — number of singular vectors to keep

        Returns
        -------
        modes : (K, N) float32 tensor — orthonormal initial modes
        """
        import numpy as np

        mu_asd = train_fc_asd.mean(axis=0)               # (N, N)
        mu_td  = train_fc_td.mean(axis=0)                # (N, N)
        delta  = mu_asd - mu_td                          # ASD-TD difference

        # SVD of the difference matrix: left singular vectors are ROI directions
        # that best explain the connectivity difference between groups.
        U, _, _ = np.linalg.svd(delta, full_matrices=True)

        K = min(num_modes, U.shape[1])
        modes = U[:, :K].T.astype(np.float32)            # (K, N)

        # If K > available singular vectors (shouldn't happen for N=200, K<<200),
        # pad with QR-orthogonalized random directions
        if num_modes > K:
            extra = np.random.randn(num_modes - K, U.shape[0]).astype(np.float32)
            for i in range(len(extra)):
                for row in modes:
                    extra[i] -= np.dot(extra[i], row) * row
                n = np.linalg.norm(extra[i])
                if n > 1e-8:
                    extra[i] /= n
            modes = np.concatenate([modes, extra], axis=0)

        return torch.from_numpy(modes)


class AdversarialConnectivityMLP(nn.Module):
    """FC-based classifier with adversarial site deconfounding (Ganin et al. 2016).

    Architecture:
        FC upper triangle (signed)
            → shared_encoder          # learns site-invariant features
            ↙                   ↘
        asd_head              grl(α) → site_head
        (minimize ASD CE)     (encoder maximises site CE via reversed grads)

    During training the encoder is pulled in two directions:
      - Minimise ASD classification loss (learn diagnosis signal)
      - Maximise site classification loss (unlearn scanner fingerprint)

    alpha is annealed 0→1 via ganin_alpha() so site deconfounding
    ramps up gradually after the ASD signal is first established.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_sites: int = 17,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Shared encoder — LazyLinear handles variable FC input size
        self.encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # ASD classification head
        self.asd_head = nn.Linear(hidden_dim, num_classes)

        # Site adversarial branch
        self.grl = GradientReversal(alpha=0.0)   # alpha set externally each epoch
        self.site_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_sites),
        )

    def forward(
        self,
        bold_windows: torch.Tensor,
        adj: torch.Tensor,
        return_site_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = ConnectivityMLPClassifier._fc_features(adj)

        features = self.encoder(x)
        asd_logits = self.asd_head(features)

        if return_site_logits:
            site_logits = self.site_head(self.grl(features))
            return asd_logits, site_logits
        return asd_logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    hidden_dim: int = 64,
    num_classes: int = 2,
    num_sites: int = 1,
    num_nodes: int = 200,
    num_modes: int = 16,
    dropout: float = 0.5,
    readout: str = "attention",
    drop_edge_p: float = 0.1,
    mode_init: "torch.Tensor | None" = None,
) -> nn.Module:
    if model_name == "graph_temporal":
        return BrainGCNClassifier(hidden_dim, num_classes, dropout, readout, drop_edge_p)
    if model_name == "gcn":
        return GraphOnlyClassifier(hidden_dim, num_classes, dropout, readout, drop_edge_p)
    if model_name == "gru":
        return TemporalGRUClassifier(hidden_dim, num_classes, dropout)
    if model_name == "fc_mlp":
        return ConnectivityMLPClassifier(hidden_dim, num_classes, dropout)
    if model_name == "adv_fc_mlp":
        return AdversarialConnectivityMLP(hidden_dim, num_classes, num_sites, dropout)
    if model_name == "brain_mode":
        return BrainModeNetwork(num_nodes, num_modes, hidden_dim, num_classes, dropout,
                                mode_init=mode_init)
    # Advanced models — lazy import to avoid circular dependency
    from brain_gcn.models.advanced_models import (
        GATClassifier, TransformerClassifier, CNN3DClassifier, GraphSAGEClassifier,
    )
    if model_name == "gat":
        return GATClassifier(hidden_dim, dropout=dropout)
    if model_name == "transformer":
        return TransformerClassifier(hidden_dim, dropout=dropout)
    if model_name == "cnn3d":
        return CNN3DClassifier(hidden_dim, dropout=dropout)
    if model_name == "graphsage":
        return GraphSAGEClassifier(hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown model_name: {model_name}")
