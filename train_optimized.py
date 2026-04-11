#!/usr/bin/env python3
"""
Complete training pipeline optimized for 95% accuracy.
Implements:
1. Proper data handling with class weighting
2. Strong regularization + data augmentation
3. Hyperparameter optimization
4. Ensemble prediction
"""
import sys
import os
from pathlib import Path
import json

# No path wrapper needed - directory structure is now flat

import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryAUROC

from brain_gcn.utils.data.datamodule import ABIDEDataModule
from brain_gcn.tasks import ClassificationTask
from brain_gcn.main import build_trainer, ensemble_predict, validate_args


def train_optimized(args):
    """Train with optimized hyperparameters."""
    pl.seed_everything(args.seed, workers=True)
    validate_args(args)
    
    print("\n" + "="*80)
    print(f"TRAINING: {args.model_name} (optimized for accuracy)")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  LR: {args.lr:.2e}, Weight decay: {args.weight_decay:.2e}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Drop edge p: {args.drop_edge_p}")
    print(f"  BOLD noise std: {args.bold_noise_std}")
    print(f"  Max epochs: {args.max_epochs}\n")
    
    # Setup data
    dm = ABIDEDataModule(
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
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup()
    
    # Compute class weights
    train_labels = [int(np.load(p, allow_pickle=True)["label"]) for p in dm._train_paths]
    labels = np.array(train_labels)
    n_td = int((labels == 0).sum())
    n_asd = int((labels == 1).sum())
    total = n_td + n_asd
    w_td = total / (2.0 * n_td) if n_td > 0 else 1.0
    w_asd = total / (2.0 * n_asd) if n_asd > 0 else 1.0
    class_weights = torch.tensor([w_td, w_asd], dtype=torch.float32)
    
    print(f"Data:")
    print(f"  Train: {len(dm._train_paths)} samples (TD={n_td}, ASD={n_asd})")
    print(f"  Val:   {len(dm._val_paths)} samples")
    print(f"  Test:  {len(dm._test_paths)} samples")
    print(f"  Class weights: TD={w_td:.2f}, ASD={w_asd:.2f}\n")
    
    # Build model
    task = ClassificationTask(
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
    )
    
    # Train
    trainer, ckpt_dir = build_trainer(args)
    print(f"Training ({args.max_epochs} epochs)...\n")
    trainer.fit(task, datamodule=dm)
    
    # Test with ensemble
    if args.test:
        print(f"\nTesting with ensemble (top-5 checkpoints)...")
        try:
            avg_probs = ensemble_predict(ckpt_dir, dm)
            preds = avg_probs.argmax(dim=-1)
            labels = torch.cat([b[2] for b in dm.test_dataloader()])
            
            acc = (preds == labels).float().mean().item()
            auc_metric = BinaryAUROC()
            auc = auc_metric(avg_probs[:, 1], labels).item()
            
            # Compute sensitivity and specificity
            tp = ((preds == 1) & (labels == 1)).float().sum()
            tn = ((preds == 0) & (labels == 0)).float().sum()
            fp = ((preds == 1) & (labels == 0)).float().sum()
            fn = ((preds == 0) & (labels == 1)).float().sum()
            
            sensitivity = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
            specificity = (tn / (tn + fp)).item() if (tn + fp) > 0 else 0.0
            
            print(f"\n{'='*80}")
            print(f"ENSEMBLE TEST RESULTS")
            print(f"{'='*80}")
            print(f"  Accuracy:    {acc*100:.2f}%")
            print(f"  AUC:         {auc:.4f}")
            print(f"  Sensitivity: {sensitivity*100:.2f}% (ASD recall)")
            print(f"  Specificity: {specificity*100:.2f}% (TD recall)")
            print(f"{'='*80}\n")
            
            # Save results
            results = {
                "model": args.model_name,
                "test_acc": acc,
                "test_auc": auc,
                "test_sensitivity": sensitivity,
                "test_specificity": specificity,
                "hyperparameters": {
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "drop_edge_p": args.drop_edge_p,
                    "bold_noise_std": args.bold_noise_std,
                }
            }
            
            results_path = Path("results") / "ensemble_results.json"
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {results_path}")
            
            if acc >= 0.95:
                print("\n🎉 SUCCESS! Achieved 95% accuracy!")
            elif acc >= 0.90:
                print("\n⚠️  Good progress (90%+ accuracy), but need more optimization")
            else:
                print("\n⏰ More work needed - try running HPO with more trials")
                
        except Exception as e:
            print(f"❌ Ensemble failed ({e}), falling back to single checkpoint")
            trainer.test(task, datamodule=dm, ckpt_path="best")
    
    return trainer, task, dm


def main():
    parser = argparse.ArgumentParser(description="Optimized Brain GCN Training")
    parser = ABIDEDataModule.add_data_specific_arguments(parser)
    parser = ClassificationTask.add_model_specific_arguments(parser)
    
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no_ensemble", action="store_true")
    
    # Override defaults with optimized values
    parser.set_defaults(
        model_name="graph_temporal",
        hidden_dim=128,
        dropout=0.4,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=16,
        drop_edge_p=0.2,
        bold_noise_std=0.02,
        cosine_t0=50,
        cosine_t_mult=2,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    
    args = parser.parse_args()
    
    if args.prepare_data:
        print("Data preparation already handled. Remove --prepare_data flag.")
        sys.exit(1)
    
    train_optimized(args)


if __name__ == "__main__":
    main()
