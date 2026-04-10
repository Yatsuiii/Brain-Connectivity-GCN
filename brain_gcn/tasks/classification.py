from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(
            model_name=model_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            readout=readout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

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
            self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_auc", self.val_auc, prog_bar=True, on_epoch=True, on_step=False)
            self.log("val_f1", self.val_f1, prog_bar=False, on_epoch=True, on_step=False)
        elif stage == "test":
            self.test_acc.update(preds, labels)
            self.test_auc.update(probs, labels)
            self.test_f1.update(preds, labels)
            self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_auc", self.test_auc, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_f1", self.test_f1, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

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
        return parser
