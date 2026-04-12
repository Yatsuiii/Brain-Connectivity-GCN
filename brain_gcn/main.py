"""
Training entry point for Brain-Connectivity-GCN.

v2 changes:
  - site_holdout as default split_strategy
  - Class weights computed from training labels → weighted CE loss
  - save_top_k=5 for checkpoint ensembling
  - ensemble_predict() utility after training
  - batch_size default lowered to 16 (site holdout = smaller train sets)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryAUROC

from brain_gcn.models.brain_gcn import BrainModeNetwork
from brain_gcn.tasks import ClassificationTask
from brain_gcn.utils.data.datamodule import ABIDEDataModule


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Skip top-5 checkpoint ensembling at test time.",
    )
    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    if args.model_name in ("fc_mlp", "adv_fc_mlp", "brain_mode") and args.use_population_adj:
        raise ValueError(
            "fc_mlp needs per-subject connectivity. Re-run with --no-use_population_adj."
        )
    if args.use_dynamic_adj_sequence and args.use_population_adj:
        raise ValueError(
            "Dynamic adjacency sequences are per-subject. Re-run with --no-use_population_adj."
        )


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_datamodule(args: argparse.Namespace) -> ABIDEDataModule:
    # fc_mlp variants need signed FC; auto-enable unless user explicitly set it
    preserve_fc_sign = getattr(args, "preserve_fc_sign", False)
    if args.model_name in ("fc_mlp", "adv_fc_mlp", "brain_mode") and not preserve_fc_sign:
        preserve_fc_sign = True

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
        preserve_fc_sign=preserve_fc_sign,
        use_fc_variance=getattr(args, "use_fc_variance", False),
        use_fisher_z=getattr(args, "use_fisher_z", False),
        n_pca_components=getattr(args, "n_pca_components", 0),
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_strategy=args.split_strategy,
        val_site=args.val_site,
        test_site=args.test_site,
        num_workers=args.num_workers,
        overwrite_cache=getattr(args, "overwrite_cache", False),
        force_prepare=args.prepare_data,
    )


def _compute_class_weights(dm: ABIDEDataModule) -> torch.Tensor:
    """Balanced class weights from training labels: total / (n_classes * n_per_class)."""
    labels = np.array(dm.train_dataset.labels)
    n_td  = int((labels == 0).sum())
    n_asd = int((labels == 1).sum())
    total = n_td + n_asd
    w_td  = total / (2.0 * n_td)
    w_asd = total / (2.0 * n_asd)
    return torch.tensor([w_td, w_asd], dtype=torch.float32)


def _discriminative_mode_init(dm: ABIDEDataModule, num_modes: int) -> torch.Tensor:
    """Load training FCs by class and compute SVD-based discriminative modes.

    Called only when model_name == 'brain_mode'. Reads the cached .npz files
    to compute (mean_FC_ASD − mean_FC_TD) and returns the top-K left singular
    vectors as the initial mode matrix (K, N).
    """
    fc_asd, fc_td = [], []
    for p in dm._train_paths:
        data = np.load(p, allow_pickle=True)
        fc   = data["mean_fc"].astype(np.float32)
        lbl  = int(data["label"])
        (fc_asd if lbl == 1 else fc_td).append(fc)

    fc_asd_arr = np.stack(fc_asd)   # (n_asd, N, N)
    fc_td_arr  = np.stack(fc_td)    # (n_td,  N, N)
    return BrainModeNetwork.discriminative_init(fc_asd_arr, fc_td_arr, num_modes)


def build_task(args: argparse.Namespace, dm: ABIDEDataModule) -> ClassificationTask:
    """Build ClassificationTask with class weights from the training split."""
    # dm.setup() must have been called before this
    try:
        class_weights = _compute_class_weights(dm)
    except Exception:
        # Fallback: no weighting (e.g. during smoke tests before full setup)
        class_weights = None

    mode_init = None
    if args.model_name == "brain_mode":
        try:
            mode_init = _discriminative_mode_init(dm, getattr(args, "num_modes", 16))
        except Exception as exc:
            print(f"[BMN] discriminative init failed ({exc}), using QR init.")

    return ClassificationTask(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        readout=args.readout,
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        bold_noise_std=args.bold_noise_std,
        drop_edge_p=args.drop_edge_p,
        cosine_t0=args.cosine_t0,
        cosine_t_mult=args.cosine_t_mult,
        cosine_eta_min=args.cosine_eta_min,
        num_sites=dm.num_sites,
        adv_site_weight=getattr(args, "adv_site_weight", 1.0),
        num_nodes=dm.num_nodes,
        num_modes=getattr(args, "num_modes", 16),
        orth_weight=getattr(args, "orth_weight", 0.01),
        mode_init=mode_init,
    )


def build_trainer(args: argparse.Namespace) -> tuple[pl.Trainer, Path]:
    ckpt_name = args.model_name
    if getattr(args, "n_pca_components", 0) > 0:
        ckpt_name += f"_pca{args.n_pca_components}"
    if args.model_name == "brain_mode":
        split_tag = getattr(args, "split_strategy", "site_holdout")[:4]  # e.g. "site" or "stra"
        ckpt_name += f"_k{getattr(args, 'num_modes', 16)}_{split_tag}"
    ckpt_dir = Path("checkpoints") / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Write run config metadata for safe ensemble verification
    config_meta = {
        "model_name": args.model_name,
        "use_dynamic_adj_sequence": args.use_dynamic_adj_sequence,
        "use_population_adj": args.use_population_adj,
    }
    config_path = ckpt_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(config_meta, f, indent=2)
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            EarlyStopping(monitor="val_auc", mode="max", patience=40),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val_auc",
                mode="max",
                save_top_k=5,             # was 1
                filename="brain-gcn-{epoch:03d}-{val_auc:.3f}",
            ),
        ],
    )
    return trainer, ckpt_dir


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

def ensemble_predict(
    ckpt_dir: str | Path,
    dm: ABIDEDataModule,
    device: str = "cpu",
) -> torch.Tensor:
    """Average softmax probabilities over the top-5 saved checkpoints.

    Verifies that each checkpoint's model config matches the datamodule's
    adjacency mode to prevent silent mismatches.

    Returns
    -------
    probs : (N_test, num_classes) averaged probability tensor
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_paths = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    # Verify config compatibility
    config_path = ckpt_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            saved_config = json.load(f)
        assert saved_config["use_dynamic_adj_sequence"] == dm.use_dynamic_adj_sequence, (
            f"Checkpoint use_dynamic_adj_sequence={saved_config['use_dynamic_adj_sequence']} "
            f"but datamodule has {dm.use_dynamic_adj_sequence}"
        )
        assert saved_config["use_population_adj"] == dm.use_population_adj, (
            f"Checkpoint use_population_adj={saved_config['use_population_adj']} "
            f"but datamodule has {dm.use_population_adj}"
        )

    all_probs: list[torch.Tensor] = []
    for ckpt in ckpt_paths:
        task = ClassificationTask.load_from_checkpoint(ckpt, map_location=device, strict=False)
        task.eval().to(device)
        batch_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dm.test_dataloader():
                bold_windows, adj = batch[0], batch[1]
                logits = task(bold_windows.to(device), adj.to(device))
                batch_probs.append(torch.softmax(logits, dim=-1).cpu())
        all_probs.append(torch.cat(batch_probs, dim=0))

    return torch.stack(all_probs).mean(0)           # (N_test, 2)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_from_args(
    args: argparse.Namespace,
) -> tuple[pl.Trainer, ClassificationTask, ABIDEDataModule]:
    pl.seed_everything(args.seed, workers=True)
    validate_args(args)

    dm = build_datamodule(args)
    # Call setup here so class weights can be computed before building the task
    dm.prepare_data()
    dm.setup()

    task = build_task(args, dm)
    trainer, ckpt_dir = build_trainer(args)
    trainer.fit(task, datamodule=dm)

    if args.test:
        if getattr(args, "no_ensemble", False):
            trainer.test(task, datamodule=dm, ckpt_path="best")
        else:
            # Ensemble over top-5 checkpoints
            try:
                avg_probs = ensemble_predict(ckpt_dir, dm)
                preds = avg_probs.argmax(dim=-1)
                # Collect ground-truth labels from test set (index 2 regardless of tuple length)
                labels = torch.cat([b[2] for b in dm.test_dataloader()])
                acc = (preds == labels).float().mean().item()
                auc_metric = BinaryAUROC()
                auc = auc_metric(avg_probs[:, 1], labels).item()
                print(f"\n[Ensemble] test_acc={acc:.4f}  test_auc={auc:.4f}")
                # Also log via trainer for experiment runner compatibility
                trainer.callback_metrics["test_acc_ensemble"] = torch.tensor(acc)
                trainer.callback_metrics["test_auc_ensemble"] = torch.tensor(auc)
            except Exception as exc:
                print(f"[Ensemble] failed ({exc}), falling back to single-best ckpt.")
                trainer.test(task, datamodule=dm, ckpt_path="best")

    return trainer, task, dm


def main() -> None:
    # RTX / Ampere+ GPUs: use TF32 for matmuls — faster with negligible precision loss
    torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    train_from_args(args)


if __name__ == "__main__":
    main()
