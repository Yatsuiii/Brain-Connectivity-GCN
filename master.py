#!/usr/bin/env python3
"""
Master script to achieve 95% accuracy.
Automatically uses synthetic data if real data not ready,
switches to real data when available.
Runs full pipeline: train → HPO → final model → ensemble test
"""
import sys
import os
from pathlib import Path
import json

# No path wrapper needed - directory structure is now flat

import argparse
import numpy as np

def check_data_readiness():
    """Check data status and recommend action."""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        return None, 0
    
    npz_files = list(processed_dir.glob("*.npz"))
    if len(npz_files) == 0:
        return None, 0
    
    # Count by site
    sites = {}
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        site = str(data.get("site", "unknown"))
        sites[site] = sites.get(site, 0) + 1
    
    is_synthetic = len(sites) == 1 and "synthetic" in sites
    return sites, is_synthetic

def print_banner(title):
    """Print formatted banner."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Master accuracy optimization pipeline for Brain GCN"
    )
    parser.add_argument("--skip_hpo", action="store_true", help="Skip HPO phase")
    parser.add_argument("--hpo_trials", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--accelerator", type=str, default="cpu")
    args = parser.parse_args()
    
    print_banner("🧠 BRAIN GCN ACCURACY OPTIMIZATION")
    
    # Check data
    print("📊 Checking data status...")
    sites, is_synthetic = check_data_readiness()
    
    if sites is None:
        print("❌ No data found. Run: python create_synthetic_data.py")
        sys.exit(1)
    
    total_subjects = sum(sites.values())
    print(f"✓ Found {total_subjects} subjects")
    print(f"✓ Sites: {sites}")
    
    if is_synthetic:
        print("⚠️  Using SYNTHETIC DATA (for quick testing)")
        data_type = "synthetic"
    else:
        print("✓ Using REAL ABIDE DATA")
        data_type = "real"
    
    print(f"\n🎯 Pipeline:")
    print(f"  1. Train initial model (optimized hyperparameters)")
    if not args.skip_hpo:
        print(f"  2. Run HPO for {args.hpo_trials} trials")
        print(f"  3. Train final model with best parameters")
    print(f"  4. Ensemble test with top-5 checkpoints")
    print(f"  5. Report accuracy (target: 95%+)")
    
    # Phase 1: Train optimized baseline
    print_banner("Phase 1: Train Optimized Baseline")
    print(f"Data: {data_type.upper()} ({total_subjects} subjects)")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Config: hidden_dim=128, dropout=0.4, LR=0.001\n")
    
    import subprocess
    cmd = [
        sys.executable,
        "train_optimized.py",
        f"--max_epochs={min(50, args.max_epochs)}",
        "--batch_size=16",
        f"--accelerator={args.accelerator}",
        "--test",
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("❌ Training failed")
        sys.exit(1)
    
    # Check initial results
    results_path = Path("results/ensemble_results.json")
    if results_path.exists():
        with open(results_path) as f:
            initial_results = json.load(f)
        initial_acc = initial_results["test_acc"]
        print(f"\n📈 Initial accuracy: {initial_acc*100:.2f}%")
        
        if initial_acc >= 0.95:
            print("🎉 ALREADY AT 95%! Optimization complete!")
            return
    
    # Phase 2: HPO (if not skipped and not synthetic)
    if not args.skip_hpo and not is_synthetic:
        print_banner(f"Phase 2: Hyperparameter Optimization ({args.hpo_trials} trials)")
        
        hpo_cmd = [
            sys.executable,
            "run_hpo.py",
            f"--n_trials={args.hpo_trials}",
            f"--max_epochs=50",
            f"--accelerator={args.accelerator}",
        ]
        
        result = subprocess.run(hpo_cmd)
        if result.returncode == 0:
            hpo_summary = Path("results/hpo_summary.json")
            if hpo_summary.exists():
                with open(hpo_summary) as f:
                    hpo_data = json.load(f)
                print(f"\n✓ HPO Best AUC: {hpo_data.get('best_value', 0):.4f}")
    
    # Phase 3: Final model on real data (if applicable)
    if is_synthetic:
        print_banner("✅ SYNTHETIC DATA TESTING COMPLETE")
        print("\nTo achieve 95% accuracy on REAL data:")
        print("1. Ensure ABIDE dataset is downloaded (data/raw/)")
        print("2. Run: python train.py --prepare_data --n_subjects 200+")
        print("3. Then run: python master.py --accelerator gpu (if available)")
    else:
        print_banner("Phase 3: Final Training with Best Parameters (Real Data)")
        final_cmd = [
            sys.executable,
            "train_optimized.py",
            f"--max_epochs={args.max_epochs}",
            "--batch_size=16",
            f"--accelerator={args.accelerator}",
            "--test",
        ]
        result = subprocess.run(final_cmd)
        
        if results_path.exists():
            with open(results_path) as f:
                final_results = json.load(f)
            final_acc = final_results["test_acc"]
            
            print_banner("🏁 FINAL RESULTS")
            print(f"Accuracy:    {final_acc*100:.2f}%")
            print(f"AUC:         {final_results.get('test_auc', 0):.4f}")
            print(f"Sensitivity: {final_results.get('test_sensitivity', 0)*100:.2f}%")
            print(f"Specificity: {final_results.get('test_specificity', 0)*100:.2f}%")
            
            if final_acc >= 0.95:
                print("\n🎉 SUCCESS! ACHIEVED 95% ACCURACY!")
            else:
                print(f"\n⚠️  Target not met. Try:")
                print("  - Increase --max_epochs")
                print("  - Use GPU: --accelerator gpu")
                print("  - Run with more subjects (200+ recommended)")

if __name__ == "__main__":
    main()
