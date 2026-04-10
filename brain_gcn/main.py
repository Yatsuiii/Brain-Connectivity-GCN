from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from brain_gcn.tasks import ClassificationTask
from brain_gcn.utils.data.datamodule import ABIDEDataModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Brain-Connectivity-GCN classifier")
    parser = ABIDEDataModule.add_data_specific_arguments(parser)
    parser = ClassificationTask.add_model_specific_arguments(parser)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.model_name == "fc_mlp" and args.use_population_adj:
        raise ValueError(
            "fc_mlp needs per-subject connectivity. Re-run with --no-use_population_adj."
        )
    if args.use_dynamic_adj_sequence and args.use_population_adj:
        raise ValueError(
            "Dynamic adjacency sequences are per-subject. Re-run with --no-use_population_adj."
        )


def build_datamodule(args: argparse.Namespace) -> ABIDEDataModule:
    return ABIDEDataModule(
        data_dir=args.data_dir,
        n_subjects=args.n_subjects,
        window_len=args.window_len,
        step=args.step,
        max_windows=args.max_windows,
        fc_threshold=args.fc_threshold,
        use_dynamic_adj=args.use_dynamic_adj,
        use_dynamic_adj_sequence=args.use_dynamic_adj_sequence,
        use_population_adj=args.use_population_adj,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_strategy=args.split_strategy,
        val_site=args.val_site,
        test_site=args.test_site,
        num_workers=args.num_workers,
        force_prepare=args.prepare_data,
    )


def build_task(args: argparse.Namespace) -> ClassificationTask:
    return ClassificationTask(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        readout=args.readout,
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_trainer(args: argparse.Namespace) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            EarlyStopping(monitor="val_auc", mode="max", patience=20),
            ModelCheckpoint(monitor="val_auc", mode="max", filename="brain-gcn-{epoch:03d}-{val_auc:.3f}"),
        ],
    )


def train_from_args(args: argparse.Namespace) -> tuple[pl.Trainer, ClassificationTask, ABIDEDataModule]:
    pl.seed_everything(args.seed, workers=True)
    validate_args(args)
    dm = build_datamodule(args)
    task = build_task(args)
    trainer = build_trainer(args)
    trainer.fit(task, datamodule=dm)
    if args.test:
        trainer.test(task, datamodule=dm, ckpt_path="best")
    return trainer, task, dm


def main() -> None:
    args = build_parser().parse_args()
    train_from_args(args)


if __name__ == "__main__":
    main()
