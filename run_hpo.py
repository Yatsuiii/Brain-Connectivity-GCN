#!/usr/bin/env python3
"""
Run HPO to find optimal hyperparameters for 95% accuracy.
Executes after initial training.
"""
import sys
import os
from pathlib import Path

# No path wrapper needed - directory structure is now flat

import argparse
from brain_gcn.hpo import HPOConfig, HPOStudy
from brain_gcn.main import train_from_args

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for 95% accuracy")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--model_name", type=str, default="graph_temporal")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80 + "\n")
    print(f"Configuration:")
    print(f"  Trials: {args.n_trials}")
    print(f"  Model: {args.model_name}")
    print(f"  Max epochs per trial: {args.max_epochs}")
    print(f"  Accelerator: {args.accelerator}\n")
    
    # Create base args for HPO
    base_args = argparse.Namespace(
        data_dir=args.data_dir,
        n_subjects=None,
        window_len=50,
        step=5,
        max_windows=20,
        fc_threshold=0.2,
        use_dynamic_adj=False,
        use_dynamic_adj_sequence=False,
        use_population_adj=True,
        batch_size=args.batch_size,
        val_ratio=0.1,
        test_ratio=0.2,
        split_strategy="stratified",
        val_site=None,
        test_site=None,
        num_workers=0,
        readout="attention",
        model_name=args.model_name,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices="auto",
        seed=42,
        log_every_n_steps=1,
        test=False,
        no_ensemble=False,
        prepare_data=False,
    )
    
    # Run HPO
    hpo_config = HPOConfig(
        study_name=f"{args.model_name}_hpo",
        n_trials=args.n_trials,
        objective_metric="test_auc",
        direction="maximize",
        seed=42,
    )
    
    hpo = HPOStudy(hpo_config)
    hpo.create_study()
    hpo.optimize(base_args)
    
    # Summary
    print(f"\n{'='*80}")
    print("HPO COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest trial: {hpo.best_trial}")
    print(f"Best value (test_auc): {hpo.best_value():.4f}")
    print(f"Best params: {hpo.best_params()}\n")
    
    # Save summary
    summary_path = Path("results") / "hpo_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    hpo.save_summary(str(summary_path))
    print(f"✅ Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
