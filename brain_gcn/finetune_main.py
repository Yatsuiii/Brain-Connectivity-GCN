"""
BC-MAE Fine-tuning Script.

Two-phase fine-tuning of a pre-trained BC-MAE encoder for ASD/TD classification.

  Phase 1 — Linear probe  (encoder frozen, ~50 epochs)
    Warms up the classification head without distorting the encoder.

  Phase 2 — Full fine-tune (encoder + head, discriminative LRs, ~150 epochs)
    Head  : lr (full)
    Encoder: lr × encoder_lr_scale (default 0.1)

Data: use_fc_degree_features=True → (W=30, N=200) mean |FC| per window,
      same feature as pre-training. Labels used only in fine-tuning loss.

Usage:
    python -m brain_gcn.finetune_main \\
        --mae_ckpt checkpoints/mae/mae-best-*.ckpt \\
        --data_dir data \\
        --probe_epochs 50 \\
        --finetune_epochs 150 \\
        --lr 5e-4
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from brain_gcn.models.mae import BrainFCClassifier, BrainFCEncoder
from brain_gcn.utils.data.datamodule import ABIDEDataModule


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class MAEClassificationTask(pl.LightningModule):
    def __init__(
        self,
        classifier: BrainFCClassifier,
        class_weights: torch.Tensor | None = None,
        lr: float = 5e-4,
        encoder_lr_scale: float = 0.1,
        weight_decay: float = 1e-4,
        bold_noise_std: float = 0.01,
        cosine_t0: int = 30,
        cosine_eta_min: float = 1e-6,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classifier", "class_weights"])
        self.model = classifier
        self.register_buffer("class_weights", class_weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.train_acc  = BinaryAccuracy()
        self.val_acc    = BinaryAccuracy()
        self.val_auc    = BinaryAUROC()
        self.val_f1     = BinaryF1Score()
        self.val_sens   = BinaryRecall()
        self.val_spec   = BinarySpecificity()
        self.test_acc   = BinaryAccuracy()
        self.test_auc   = BinaryAUROC()
        self.test_f1    = BinaryF1Score()
        self.test_sens  = BinaryRecall()
        self.test_spec  = BinarySpecificity()

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, adj)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, adj, labels, _ = batch
        # Spatial BC-MAE: adj = (B, N, N) full FC matrix = N ROI tokens × N-dim features
        x = adj
        if self.hparams.bold_noise_std > 0.0:
            sig = x.std(dim=(1, 2), keepdim=True).detach()
            x = x + torch.randn_like(x) * self.hparams.bold_noise_std * sig
        logits = self(x)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(-1)
        self.train_acc.update(preds, labels)
        self.log("train_loss", loss,           prog_bar=True,  on_epoch=True, on_step=False)
        self.log("train_acc",  self.train_acc, prog_bar=True,  on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, adj, labels, _ = batch
        x = adj  # (B, N, N) full FC matrix
        logits = self(x)
        loss   = self.loss_fn(logits, labels)
        probs  = torch.softmax(logits, -1)[:, 1]
        preds  = logits.argmax(-1)
        self.val_acc.update(preds, labels)
        self.val_auc.update(probs, labels)
        self.val_f1.update(preds, labels)
        self.val_sens.update(preds, labels)
        self.val_spec.update(preds, labels)
        self.log("val_loss", loss,           prog_bar=True,  on_epoch=True, on_step=False)
        self.log("val_acc",  self.val_acc,   prog_bar=True,  on_epoch=True, on_step=False)
        self.log("val_auc",  self.val_auc,   prog_bar=True,  on_epoch=True, on_step=False)
        self.log("val_f1",   self.val_f1,    prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_sens", self.val_sens,  prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_spec", self.val_spec,  prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, adj, labels, _ = batch
        x = adj  # (B, N, N) full FC matrix
        logits = self(x)
        loss   = self.loss_fn(logits, labels)
        probs  = torch.softmax(logits, -1)[:, 1]
        preds  = logits.argmax(-1)
        self.test_acc.update(preds, labels)
        self.test_auc.update(probs, labels)
        self.test_f1.update(preds, labels)
        self.test_sens.update(preds, labels)
        self.test_spec.update(preds, labels)
        self.log("test_loss", loss,            on_epoch=True, on_step=False)
        self.log("test_acc",  self.test_acc,   prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_auc",  self.test_auc,   prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_f1",   self.test_f1,    prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_sens", self.test_sens,  prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_spec", self.test_spec,  prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        enc_ids  = {id(p) for p in self.model.encoder.parameters()}
        enc_params  = [p for p in self.model.parameters() if id(p) in enc_ids]
        head_params = [p for p in self.model.parameters() if id(p) not in enc_ids]

        if self.hparams.freeze_encoder:
            param_groups = [{"params": head_params, "lr": self.hparams.lr}]
        else:
            param_groups = [
                {"params": head_params, "lr": self.hparams.lr},
                {"params": enc_params,  "lr": self.hparams.lr * self.hparams.encoder_lr_scale},
            ]

        opt = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=self.hparams.cosine_t0,
            eta_min=self.hparams.cosine_eta_min,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_class_weights(dm: ABIDEDataModule) -> torch.Tensor:
    labels = np.array([int(np.load(p, allow_pickle=True)["label"]) for p in dm._train_paths])
    n_td  = int((labels == 0).sum())
    n_asd = int((labels == 1).sum())
    total = n_td + n_asd
    return torch.tensor([total / (2.0 * n_td), total / (2.0 * n_asd)], dtype=torch.float32)


def _load_encoder(
    ckpt_path: str,
    num_rois: int,
    num_windows: int,
    hidden_dim: int,
    num_heads: int,
    encoder_layers: int,
    dropout: float,
) -> BrainFCEncoder:
    """Extract BrainFCEncoder weights from a BrainMAETask Lightning checkpoint."""
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    enc_state = {
        k[len("mae.encoder."):]: v
        for k, v in state.items()
        if k.startswith("mae.encoder.")
    }
    if not enc_state:
        raise KeyError(
            f"No 'mae.encoder.*' keys found in {ckpt_path}. "
            "Make sure you pass a BrainMAETask checkpoint, not a classifier checkpoint."
        )

    encoder = BrainFCEncoder(
        num_rois=num_rois,
        num_windows=num_windows,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=encoder_layers,
        dropout=dropout,
    )
    encoder.load_state_dict(enc_state, strict=True)
    print(f"Loaded encoder from {ckpt_path}  ({len(enc_state)} tensors)")
    return encoder


def _load_head_weights(task: MAEClassificationTask, ckpt_path: str) -> None:
    """Restore time_attn + head weights from a previous phase checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    mapping = {}
    for k, v in sd.items():
        if k.startswith("model.time_attn.") or k.startswith("model.head."):
            new_k = k[len("model."):]
            mapping[new_k] = v
    if mapping:
        current = task.model.state_dict()
        current.update(mapping)
        task.model.load_state_dict(current, strict=True)
        print(f"Restored {len(mapping)} head tensors from {ckpt_path}")


def _make_trainer(
    max_epochs: int,
    ckpt_dir: Path,
    prefix: str,
    accelerator: str,
    devices: str,
    patience: int = 30,
) -> pl.Trainer:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_auc", mode="max", patience=patience),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val_auc",
                mode="max",
                save_top_k=3,
                filename=f"{prefix}-{{epoch:03d}}-{{val_auc:.3f}}",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BC-MAE Fine-tuning")
    p.add_argument("--mae_ckpt", type=str, required=True,
                   help="Path to best MAE pre-training checkpoint (.ckpt)")
    p.add_argument("--data_dir",         type=str,   default="data")
    p.add_argument("--max_windows",      type=int,   default=30)
    p.add_argument("--hidden_dim",       type=int,   default=128)
    p.add_argument("--num_heads",        type=int,   default=4)
    p.add_argument("--encoder_layers",   type=int,   default=4)
    p.add_argument("--dropout_encoder",  type=float, default=0.1)
    p.add_argument("--dropout_head",     type=float, default=0.5)
    # Phase 1
    p.add_argument("--probe_epochs",     type=int,   default=50,
                   help="Epochs with frozen encoder (linear probe).")
    p.add_argument("--probe_lr",         type=float, default=1e-3)
    # Phase 2
    p.add_argument("--finetune_epochs",  type=int,   default=150,
                   help="Epochs with full encoder fine-tuning.")
    p.add_argument("--finetune_lr",      type=float, default=5e-4)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="Encoder LR = finetune_lr × this. Default 0.1 (10x smaller).")
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--bold_noise_std",   type=float, default=0.01)
    p.add_argument("--cosine_t0",        type=int,   default=30)
    p.add_argument("--cosine_eta_min",   type=float, default=1e-6)
    # Data
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--split_strategy",   choices=["stratified", "site_holdout"],
                   default="stratified")
    p.add_argument("--val_site",         type=str,   default=None)
    p.add_argument("--test_site",        type=str,   default=None)
    # Misc
    p.add_argument("--accelerator",      type=str,   default="auto")
    p.add_argument("--devices",          type=str,   default="auto")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--ckpt_dir",         type=str,   default="checkpoints/mae_finetune")
    p.add_argument("--test",             action="store_true",
                   help="Run test set evaluation after fine-tuning.")
    p.add_argument("--skip_probe",       action="store_true",
                   help="Skip Phase 1 and jump straight to full fine-tuning.")
    return p


def main() -> None:
    torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    # ── Data ────────────────────────────────────────────────────────────
    # Spatial BC-MAE uses the full mean FC matrix (N, N) as input.
    # With use_population_adj=False and preserve_fc_sign=True, each subject's
    # adj = (N, N) signed mean FC — exactly what the spatial encoder expects.
    dm = ABIDEDataModule(
        data_dir=args.data_dir,
        use_population_adj=False,
        preserve_fc_sign=True,      # signed FC → adj = (N, N) mean FC per subject
        fc_threshold=0.0,           # no thresholding — matches pre-training distribution
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_strategy=args.split_strategy,
        val_site=args.val_site,
        test_site=args.test_site,
    )
    dm.prepare_data()
    dm.setup()

    num_rois      = dm.num_nodes
    class_weights = _compute_class_weights(dm)
    print(f"num_rois={num_rois}  class_weights={class_weights.tolist()}")

    # ── Load pre-trained encoder ─────────────────────────────────────────
    encoder = _load_encoder(
        ckpt_path=args.mae_ckpt,
        num_rois=num_rois,
        num_windows=num_rois,   # spatial MAE: num_windows = num_rois (200)
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        dropout=args.dropout_encoder,
    )

    ckpt_dir = Path(args.ckpt_dir)

    best_probe_ckpt: str | None = None

    # ── Phase 1: Linear probe (encoder frozen) ───────────────────────────
    if not args.skip_probe:
        print(f"\n{'='*60}")
        print(f"Phase 1: Linear probe  ({args.probe_epochs} epochs, LR={args.probe_lr})")
        print(f"{'='*60}")

        classifier_p1 = BrainFCClassifier(
            encoder=encoder,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            dropout=args.dropout_head,
            freeze_encoder=True,
        )
        task_p1 = MAEClassificationTask(
            classifier=classifier_p1,
            class_weights=class_weights,
            lr=args.probe_lr,
            encoder_lr_scale=0.0,   # ignored while frozen
            weight_decay=args.weight_decay,
            bold_noise_std=0.0,     # no augmentation during probe
            cosine_t0=args.cosine_t0,
            cosine_eta_min=args.cosine_eta_min,
            freeze_encoder=True,
        )
        trainer_p1 = _make_trainer(
            max_epochs=args.probe_epochs,
            ckpt_dir=ckpt_dir / "probe",
            prefix="probe",
            accelerator=args.accelerator,
            devices=args.devices,
            patience=20,
        )
        trainer_p1.fit(task_p1, datamodule=dm)
        best_probe_ckpt = trainer_p1.checkpoint_callback.best_model_path
        best_probe_auc  = trainer_p1.callback_metrics.get("val_auc", torch.tensor(0.0))
        print(f"Phase 1 best val_auc: {float(best_probe_auc):.4f}")

    # ── Phase 2: Full fine-tuning ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Full fine-tune ({args.finetune_epochs} epochs, "
          f"LR={args.finetune_lr}, enc_scale={args.encoder_lr_scale})")
    print(f"{'='*60}")

    classifier_p2 = BrainFCClassifier(
        encoder=copy.deepcopy(encoder),
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout_head,
        freeze_encoder=False,
    )
    task_p2 = MAEClassificationTask(
        classifier=classifier_p2,
        class_weights=class_weights,
        lr=args.finetune_lr,
        encoder_lr_scale=args.encoder_lr_scale,
        weight_decay=args.weight_decay,
        bold_noise_std=args.bold_noise_std,
        cosine_t0=args.cosine_t0,
        cosine_eta_min=args.cosine_eta_min,
        freeze_encoder=False,
    )

    # Transfer warmed-up head weights from Phase 1
    if best_probe_ckpt:
        _load_head_weights(task_p2, best_probe_ckpt)

    trainer_p2 = _make_trainer(
        max_epochs=args.finetune_epochs,
        ckpt_dir=ckpt_dir / "finetune",
        prefix="ft",
        accelerator=args.accelerator,
        devices=args.devices,
        patience=40,
    )
    trainer_p2.fit(task_p2, datamodule=dm)
    best_ft_auc = trainer_p2.callback_metrics.get("val_auc", torch.tensor(0.0))
    print(f"\nPhase 2 best val_auc: {float(best_ft_auc):.4f}")

    if args.test:
        print("\nRunning test set evaluation ...")
        trainer_p2.test(task_p2, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
