"""
PyTorch Lightning training task for ASD/TD classification.

v2 changes:
  - class_weights arg → weighted CrossEntropyLoss (fixes class imbalance)
  - CosineAnnealingWarmRestarts scheduler (T_0=50, T_mult=2)
  - BOLD noise augmentation in training_step
  - Sensitivity (ASD recall) + Specificity (TD recall) metrics added
  - drop_edge_p forwarded to build_model
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from brain_gcn.models import build_model


class ClassificationTask(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        readout: str = "attention",
        model_name: str = "graph_temporal",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        bold_noise_std: float = 0.01,
        drop_edge_p: float = 0.1,
        cosine_t0: int = 50,
        cosine_t_mult: int = 2,
        cosine_eta_min: float = 1e-5,
    ):
        """
        Parameters
        ----------
        class_weights   : 1-D tensor of length num_classes for weighted CE.
                          Typically [total/(2*n_td), total/(2*n_asd)].
                          None = unweighted (original behaviour).
        bold_noise_std  : std dev of Gaussian noise added to bold_windows
                          during training. 0.0 disables augmentation.
        drop_edge_p     : edge drop probability forwarded to graph models.
        cosine_t0       : CosineAnnealingWarmRestarts first restart epoch.
        cosine_t_mult   : restart interval multiplier.
        cosine_eta_min  : minimum LR after annealing.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        # Store class_weights separately (tensors don't serialise cleanly via hparams)
        self.register_buffer("class_weights", class_weights)

        self.model = build_model(
            model_name=model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            readout=readout,
            drop_edge_p=drop_edge_p,
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # --- Metrics --------------------------------------------------------
        self.train_acc = BinaryAccuracy()

        self.val_acc  = BinaryAccuracy()
        self.val_auc  = BinaryAUROC()
        self.val_f1   = BinaryF1Score()
        self.val_sens = BinaryRecall()          # sensitivity = ASD recall
        self.val_spec = BinarySpecificity()     # specificity = TD recall

        self.test_acc  = BinaryAccuracy()
        self.test_auc  = BinaryAUROC()
        self.test_f1   = BinaryF1Score()
        self.test_sens = BinaryRecall()
        self.test_spec = BinarySpecificity()

    # ------------------------------------------------------------------
    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.model(bold_windows, adj)

    def _step(self, batch, stage: str) -> torch.Tensor:
        bold_windows, adj, labels = batch
        logits = self(bold_windows, adj)
        loss = self.loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        if stage == "train":
            self.train_acc.update(preds, labels)
            self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)

        elif stage == "val":
            self.val_acc.update(preds, labels)
            self.val_auc.update(probs, labels)
            self.val_f1.update(preds, labels)
            self.val_sens.update(preds, labels)
            self.val_spec.update(preds, labels)
            self.log("val_acc",  self.val_acc,  prog_bar=True,  on_epoch=True, on_step=False)
            self.log("val_auc",  self.val_auc,  prog_bar=True,  on_epoch=True, on_step=False)
            self.log("val_f1",   self.val_f1,   prog_bar=False, on_epoch=True, on_step=False)
            self.log("val_sens", self.val_sens, prog_bar=False, on_epoch=True, on_step=False)
            self.log("val_spec", self.val_spec, prog_bar=False, on_epoch=True, on_step=False)

        elif stage == "test":
            self.test_acc.update(preds, labels)
            self.test_auc.update(probs, labels)
            self.test_f1.update(preds, labels)
            self.test_sens.update(preds, labels)
            self.test_spec.update(preds, labels)
            self.log("test_acc",  self.test_acc,  prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_auc",  self.test_auc,  prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_f1",   self.test_f1,   prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_sens", self.test_sens, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_spec", self.test_spec, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        bold_windows, adj, labels = batch
        # Relative BOLD noise augmentation (training only)
        # Noise std is proportional to per-sample signal std for consistent augmentation
        if self.hparams.bold_noise_std > 0.0:
            signal_std = bold_windows.std(dim=(1, 2), keepdim=True).detach()
            noise = torch.randn_like(bold_windows) * self.hparams.bold_noise_std * signal_std
            bold_windows = bold_windows + noise
        return self._step((bold_windows, adj, labels), "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=self.hparams.cosine_t0,
            T_mult=self.hparams.cosine_t_mult,
            eta_min=self.hparams.cosine_eta_min,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    @staticmethod
    def add_model_specific_arguments(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--readout", choices=["mean", "attention"], default="attention")
        parser.add_argument(
            "--model_name",
            choices=["graph_temporal", "gcn", "gru", "fc_mlp"],
            default="graph_temporal",
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--bold_noise_std", type=float, default=0.01)
        parser.add_argument("--drop_edge_p", type=float, default=0.1)
        parser.add_argument("--cosine_t0", type=int, default=50)
        parser.add_argument("--cosine_t_mult", type=int, default=2,
                           help="CosineAnnealingWarmRestarts restart interval multiplier")
        parser.add_argument("--cosine_eta_min", type=float, default=1e-5,
                           help="CosineAnnealingWarmRestarts minimum learning rate")
        return parser
