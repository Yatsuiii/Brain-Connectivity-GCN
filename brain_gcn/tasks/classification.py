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
from brain_gcn.utils.grl import ganin_alpha


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
        num_sites: int = 1,
        adv_site_weight: float = 1.0,
        num_nodes: int = 200,
        num_modes: int = 16,
        orth_weight: float = 0.01,
        mode_init: "torch.Tensor | None" = None,
        in_features: int = 1,
    ):
        """
        Parameters
        ----------
        class_weights    : 1-D tensor of length num_classes for weighted CE.
        bold_noise_std   : std dev of Gaussian noise added during training.
        drop_edge_p      : edge drop probability for graph models.
        cosine_t0        : CosineAnnealingWarmRestarts first restart epoch.
        cosine_t_mult    : restart interval multiplier.
        cosine_eta_min   : minimum LR after annealing.
        num_sites        : number of acquisition sites (for adv_fc_mlp).
        adv_site_weight  : weight on the adversarial site loss term.
        in_features      : node feature dimension (1 for BOLD std, N for FC rows).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "mode_init"])
        self.register_buffer("class_weights", class_weights)

        self.model = build_model(
            model_name=model_name,
            hidden_dim=hidden_dim,
            num_sites=num_sites,
            num_nodes=num_nodes,
            num_modes=num_modes,
            dropout=dropout,
            readout=readout,
            drop_edge_p=drop_edge_p,
            mode_init=mode_init,
            in_features=in_features,
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        # Site cross-entropy — unweighted (sites roughly balanced)
        self.site_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

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

    @property
    def _is_adversarial(self) -> bool:
        return self.hparams.model_name in ("adv_fc_mlp", "adv_brain_mode")

    # ------------------------------------------------------------------
    def forward(self, bold_windows: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.model(bold_windows, adj)

    def _step(self, batch, stage: str) -> torch.Tensor:
        bold_windows, adj, labels, site_ids = batch
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
        bold_windows, adj, labels, site_ids = batch
        if self.hparams.bold_noise_std > 0.0:
            signal_std = bold_windows.std(dim=(1, 2), keepdim=True).detach()
            noise = torch.randn_like(bold_windows) * self.hparams.bold_noise_std * signal_std
            bold_windows = bold_windows + noise

        if self._is_adversarial:
            # Dual loss: ASD classification + adversarial site deconfounding
            asd_logits, site_logits = self.model(
                bold_windows, adj, return_site_logits=True
            )
            asd_loss  = self.loss_fn(asd_logits, labels)
            site_loss = self.site_loss_fn(site_logits, site_ids)
            loss = asd_loss + self.hparams.adv_site_weight * site_loss

            probs = torch.softmax(asd_logits, dim=-1)[:, 1]
            preds = torch.argmax(asd_logits, dim=-1)

            self.log("train_asd_loss",  asd_loss,  prog_bar=False, on_epoch=True, on_step=False)
            self.log("train_site_loss", site_loss, prog_bar=False, on_epoch=True, on_step=False)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            self.train_acc.update(preds, labels)
            self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        else:
            loss = self._step((bold_windows, adj, labels, site_ids), "train")

        # Orthogonality regularization — BMN only (model exposes orthogonality_loss())
        if hasattr(self.model, "orthogonality_loss") and self.hparams.orth_weight > 0.0:
            orth = self.model.orthogonality_loss()
            loss = loss + self.hparams.orth_weight * orth
            self.log("train_orth_loss", orth, prog_bar=False, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_start(self) -> None:
        """Anneal the GRL alpha at the start of each epoch."""
        if self._is_adversarial:
            alpha = ganin_alpha(self.current_epoch, self.trainer.max_epochs)
            self.model.grl.alpha = alpha
            self.log("grl_alpha", alpha, prog_bar=False, on_epoch=True, on_step=False)

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
            choices=["graph_temporal", "gcn", "gru", "fc_mlp", "adv_fc_mlp",
                     "gat", "transformer", "cnn3d", "graphsage",
                     "brain_mode", "adv_brain_mode", "dynamic_fc_attn"],
            default="graph_temporal",
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--adv_site_weight", type=float, default=1.0,
                            help="Weight on adversarial site loss (adv_fc_mlp only).")
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--bold_noise_std", type=float, default=0.01)
        parser.add_argument("--drop_edge_p", type=float, default=0.1)
        parser.add_argument("--cosine_t0", type=int, default=50)
        parser.add_argument("--cosine_t_mult", type=int, default=2,
                           help="CosineAnnealingWarmRestarts restart interval multiplier")
        parser.add_argument("--cosine_eta_min", type=float, default=1e-5,
                           help="CosineAnnealingWarmRestarts minimum learning rate")
        parser.add_argument("--num_modes", type=int, default=16,
                           help="Brain Mode Network: number of learnable modes K")
        parser.add_argument("--orth_weight", type=float, default=0.01,
                           help="Brain Mode Network: orthogonality regularization weight")
        return parser
