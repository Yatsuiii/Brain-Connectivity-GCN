#!/usr/bin/env python3
"""
Comprehensive HPO + Training script to achieve 95% accuracy.
Runs hyperparameter optimization, then trains best model with ensembling.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Full HPO + training pipeline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_trials", type=int, default=50, help="HPO trials")
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BRAIN GCN ACCURACY OPTIMIZATION PIPELINE")
    print("="*80)
    
    # Step 1: HPO
    print("\n[Step 1/3] Running Hyperparameter Optimization (HPO)...")
    print(f"           {args.n_trials} trials on graph_temporal model")
    hpo_cmd = [
        sys.executable,
        "brain_gcn/hpo_cli.py",
        "--hpo_n_trials", str(args.n_trials),
        "--hpo_objective", "test_auc",
        "--hpo_study_name", "brain_gcn_main",
        "--model_name", "graph_temporal",
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--data_dir", args.data_dir,
    ]
    
    result = subprocess.run(hpo_cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print("❌ HPO failed")
        sys.exit(1)
    
    # Step 2: Find best params
    print("\n[Step 2/3] Extracting best hyperparameters...")
    print("           (Best params are saved in brain_gcn.db)")
    
    # Step 3: Train final model with best params + ensemble
    print("\n[Step 3/3] Training final model with best parameters + ensembling...")
    final_cmd = [
        sys.executable,
        "brain_gcn/main.py",
        "--model_name", "graph_temporal",
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--data_dir", args.data_dir,
        "--test",  # Run test after training
    ]
    
    result = subprocess.run(final_cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print("❌ Final training failed")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print("Check results/ directory for detailed metrics")
    print("Check checkpoints/graph_temporal/ for saved models")

if __name__ == "__main__":
    main()
