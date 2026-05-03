"""
Brain Connectivity Masked Autoencoder (BC-MAE).

Architecture (He et al. MAE 2022, adapted for temporal FC windows):

  Pre-training
  ─────────────
  Input  : (B, W, N) — per-window ROI connectivity strengths (mean |FC| per window)
  Mask   : random 50% of W windows are hidden
  Encoder: Transformer on visible windows only  →  (B, W_vis, H)
  Decoder: Lightweight Transformer on all positions (visible + mask tokens)
            →  reconstruction head  →  (B, W, N)
  Loss   : MSE on masked windows only

  Fine-tuning
  ────────────
  Encoder (loaded from pre-training, optionally frozen)
           +  attention pooling over all W windows
           +  MLP classifier  →  (B, 2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------

class BrainFCEncoder(nn.Module):
    """Transformer encoder operating on visible FC windows.

    Each time window's ROI connectivity profile (N-dim) is treated as a
    "patch" — analogous to image patches in ViT/MAE.
    """

    def __init__(
        self,
        num_rois: int = 200,
        num_windows: int = 30,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project each window's ROI features to hidden dim
        self.patch_embed = nn.Linear(num_rois, hidden_dim)

        # Learnable positional embedding — one per window position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_windows, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, W_visible, N)  visible windows
        ids_keep : (B, W_visible)     original positions of visible windows
        """
        B, W_vis, N = x.shape

        # Project patches
        x = self.patch_embed(x)   # (B, W_vis, H)

        # Add positional embeddings at the original positions
        if ids_keep is not None:
            pos = self.pos_embed.expand(B, -1, -1)                     # (B, W_all, H)
            pos_vis = torch.gather(
                pos, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim) # (B, W_vis, H)
            )
        else:
            pos_vis = self.pos_embed[:, :W_vis, :]

        x = x + pos_vis
        x = self.norm(self.transformer(x))
        return x                   # (B, W_vis, H)


# ---------------------------------------------------------------------------
# MAE (pre-training)
# ---------------------------------------------------------------------------

class BrainMAE(nn.Module):
    """Masked Autoencoder for brain FC windows."""

    def __init__(
        self,
        num_rois: int = 200,
        num_windows: int = 30,
        hidden_dim: int = 128,
        decoder_dim: int = 64,
        num_heads: int = 4,
        encoder_layers: int = 4,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_windows = num_windows
        self.num_rois    = num_rois
        self.mask_ratio  = mask_ratio
        self.hidden_dim  = hidden_dim
        self.decoder_dim = decoder_dim

        # Encoder (shared with fine-tuning)
        self.encoder = BrainFCEncoder(
            num_rois=num_rois,
            num_windows=num_windows,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        # Project encoder output to decoder dim
        self.enc_to_dec = nn.Linear(hidden_dim, decoder_dim, bias=False)

        # Learnable mask token (broadcast across masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (all W positions)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_windows, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Lightweight decoder
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=max(1, decoder_dim // 32),
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Reconstruction head: predict ROI connectivity for each window
        self.recon_head = nn.Linear(decoder_dim, num_rois)

    # ------------------------------------------------------------------
    def _random_masking(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask windows. Returns visible subset, binary mask, restore indices."""
        B, W, _ = x.shape
        num_keep = int(W * (1 - self.mask_ratio))

        # Random shuffle per sample
        noise       = torch.rand(B, W, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]                          # (B, num_keep)
        x_vis    = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])       # (B, num_keep, N)
        )

        # Binary mask: 1 = masked, 0 = visible
        mask = torch.ones(B, W, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_vis, mask, ids_restore, ids_keep

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pre-training.

        Returns
        -------
        loss      : scalar MSE on masked windows
        mask      : (B, W) binary mask (1=masked) for logging
        """
        B, W, N = x.shape

        # Mask
        x_vis, mask, ids_restore, ids_keep = self._random_masking(x)

        # Encode visible
        enc = self.encoder(x_vis, ids_keep=ids_keep)  # (B, num_keep, H)
        enc = self.enc_to_dec(enc)                    # (B, num_keep, D)

        # Decode: reconstruct all W positions
        # Fill masked positions with mask token
        num_keep  = enc.shape[1]
        num_mask  = W - num_keep
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        # Concatenate visible encoded + mask tokens, then unshuffle
        full = torch.cat([enc, mask_tokens], dim=1)  # (B, W, D)
        full = torch.gather(
            full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )

        # Add decoder positional embeddings and decode
        full = full + self.decoder_pos_embed
        dec  = self.decoder_norm(self.decoder(full))  # (B, W, D)

        # Reconstruct
        pred = self.recon_head(dec)  # (B, W, N)

        # MSE loss on masked windows only
        loss = (pred - x).pow(2).mean(dim=-1)  # (B, W)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss, mask

    def encode_all(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all W windows (no masking) for downstream tasks."""
        return self.encoder(x)  # (B, W, H)


# ---------------------------------------------------------------------------
# Fine-tuning classifier
# ---------------------------------------------------------------------------

class BrainFCClassifier(nn.Module):
    """ASD/TD classifier with pre-trained BC-MAE encoder.

    Encoder can be frozen (linear probing) or fine-tuned end-to-end.
    """

    def __init__(
        self,
        encoder: BrainFCEncoder,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        H = hidden_dim
        # Attention pooling over time: which windows discriminate ASD?
        self.time_attn = nn.Linear(H, 1)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None = None,  # kept for interface compatibility
    ) -> torch.Tensor:
        # x: (B, W, N)
        if self.freeze_encoder:
            with torch.no_grad():
                enc = self.encoder(x)     # (B, W, H)
        else:
            enc = self.encoder(x)

        # Attention-weighted pooling over time
        attn   = torch.softmax(self.time_attn(enc).squeeze(-1), dim=1)  # (B, W)
        pooled = (enc * attn.unsqueeze(-1)).sum(dim=1)                  # (B, H)

        return self.head(pooled)

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self.freeze_encoder = False
