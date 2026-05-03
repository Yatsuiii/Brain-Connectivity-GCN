"""
BC-MAE Pre-training Script.

Self-supervised pre-training on ALL ABIDE subjects (no labels needed).

Input per subject: (W=30, N=200) mean |FC| per ROI per window
  - Loaded from fc_windows.npz, site-corrected, then mean |FC| per window
  - Same feature as --use_fc_degree_features in the classification pipeline

Task: BrainMAE masks 50% of windows, reconstructs them from visible ones.
Loss: MSE on masked windows only.

Saves: checkpoints/mae/mae-best-*.ckpt  (full BrainMAETask checkpoint)

Usage:
    python -m brain_gcn.pretrain_main \\
        --data_dir data \\
        --max_epochs 200 \\
        --hidden_dim 128 \\
        --lr 1e-3

Then fine-tune with:
    python -m brain_gcn.finetune_main \\
        --mae_ckpt checkpoints/mae/mae-best-*.ckpt \\
        --data_dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from brain_gcn.models.mae import BrainMAE


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MAEDataset(Dataset):
    """All ABIDE subjects → (N, N) full FC matrix for spatial BC-MAE pre-training.

    Each subject is represented as N=200 tokens, where token i is ROI i's full
    connectivity profile (its FC row). The MAE masks 50% of ROIs and reconstructs
    their FC rows — forcing the encoder to learn which ROIs co-activate.
    """

    def __init__(
        self,
        npz_dir: str | Path,
        site_fc_mean: dict[str, np.ndarray] | None = None,
    ):
        self.paths = sorted(Path(npz_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found in {npz_dir}")
        self.site_fc_mean = site_fc_mean or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = np.load(self.paths[idx], allow_pickle=True)
        site = str(data["site"])

        fc = data["mean_fc"].astype(np.float32)   # (N, N)
        if site in self.site_fc_mean:
            fc = fc - self.site_fc_mean[site]

        return torch.FloatTensor(fc)   # (N, N) — each row i = ROI i's FC profile


def _compute_site_fc_mean(npz_dir: Path) -> dict[str, np.ndarray]:
    """Per-site mean FC matrix (N, N) across all subjects (no train/test split
    needed here since pre-training is fully self-supervised)."""
    site_sums: dict[str, np.ndarray] = {}
    site_counts: dict[str, int] = {}
    for p in sorted(npz_dir.glob("*.npz")):
        data = np.load(p, allow_pickle=True)
        site = str(data["site"])
        fc = data["mean_fc"].astype(np.float32)
        if site not in site_sums:
            site_sums[site] = np.zeros_like(fc)
            site_counts[site] = 0
        site_sums[site] += fc
        site_counts[site] += 1
    return {s: site_sums[s] / site_counts[s] for s in site_sums}


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class BrainMAETask(pl.LightningModule):
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
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mae = BrainMAE(
            num_rois=num_rois,
            num_windows=num_windows,
            hidden_dim=hidden_dim,
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            mask_ratio=mask_ratio,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _ = self.mae(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, _ = self.mae(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def _lr_lambda(epoch: int) -> float:
            wu = self.hparams.warmup_epochs
            if epoch < wu:
                return epoch / max(1, wu)
            progress = (epoch - wu) / max(1, self.hparams.max_epochs - wu)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BC-MAE Pre-training")
    p.add_argument("--data_dir",       type=str,   default="data")
    p.add_argument("--max_windows",    type=int,   default=30)
    p.add_argument("--max_epochs",     type=int,   default=200)
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--decoder_dim",    type=int,   default=64)
    p.add_argument("--num_heads",      type=int,   default=4)
    p.add_argument("--encoder_layers", type=int,   default=4)
    p.add_argument("--decoder_layers", type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--mask_ratio",     type=float, default=0.5)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--warmup_epochs",  type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--val_ratio",      type=float, default=0.1)
    p.add_argument("--accelerator",    type=str,   default="auto")
    p.add_argument("--devices",        type=str,   default="auto")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--ckpt_dir",       type=str,   default="checkpoints/mae")
    return p


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    processed_dir = Path(args.data_dir) / "processed"
    print(f"Computing site FC means from {processed_dir} ...")
    site_fc_mean = _compute_site_fc_mean(processed_dir)
    print(f"  {len(site_fc_mean)} sites found.")

    full_ds = MAEDataset(processed_dir, site_fc_mean=site_fc_mean)
    n = len(full_ds)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    rng = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=rng)
    print(f"Pre-training split: {n_train} train / {n_val} val  ({n} total)")

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)

    first = np.load(full_ds.paths[0], allow_pickle=True)
    num_rois = int(first["mean_fc"].shape[0])
    # Spatial MAE: each of the N ROIs is a "window", its FC row (N-dim) is the patch feature
    num_windows = num_rois
    print(f"Spatial BC-MAE: {num_rois} ROIs × {num_rois}-dim FC rows")

    task = BrainMAETask(
        num_rois=num_rois,
        num_windows=num_windows,  # = num_rois (200) — spatial MAE
        hidden_dim=args.hidden_dim,
        decoder_dim=args.decoder_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=30),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="mae-best-{epoch:03d}-{val_loss:.4f}",
            ),
        ],
    )

    trainer.fit(task, train_dl, val_dl)
    best = trainer.checkpoint_callback.best_model_path
    print(f"\nPre-training complete.")
    print(f"Best checkpoint: {best}")
    print(f"\nNext step:")
    print(f"  python -m brain_gcn.finetune_main --mae_ckpt {best} --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
