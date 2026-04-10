"""
Quick start examples for experiment management and evaluation.

Run these scripts to learn the new functionality.
"""

import argparse
from pathlib import Path

from brain_gcn.main import build_parser, train_from_args
from brain_gcn.hpo import HPOConfig, HPOStudy
from brain_gcn.ablation import AblationStudy
from brain_gcn.utils.cross_validation import kfold_cross_validate
from brain_gcn.utils.tracking import ExperimentTracker, RunLogger
from brain_gcn.utils.evaluation import compute_metrics, compute_roc_curve
import numpy as np
import torch


def example_1_basic_training():
    """Example 1: Train a model with tracking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Train Model with Experiment Tracking")
    print("=" * 70)

    # Create parser and parse minimal args
    parser = build_parser()
    args = parser.parse_args([
        "--model_name", "graph_temporal",
        "--max_epochs", "1",  # Smoke test
        "--max_windows", "5",
        "--batch_size", "4",
        "--accelerator", "cpu",
    ])

    # Create experiment tracker
    tracker = ExperimentTracker(output_dir="experiments/demo")

    # Train with logging
    with RunLogger("demo_run", args, tracker) as metadata:
        trainer, task, dm = train_from_args(args)
        
        # Simulate metrics (normally from trainer)
        metrics = {
            "test_auc": 0.75,
            "test_acc": 0.70,
        }
        metadata.update_metrics(metrics)
        metadata.set_checkpoint_path("checkpoints/demo.ckpt")

    tracker.save_summary()
    print("✓ Run logged to experiments/demo/summary.json")


def example_2_ablation_study():
    """Example 2: Run ablation study."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Ablation Study - Component Importance")
    print("=" * 70)

    # Create parser and parse minimal args
    parser = build_parser()
    args = parser.parse_args([
        "--model_name", "graph_temporal",
        "--max_epochs", "1",
        "--batch_size", "4",
        "--accelerator", "cpu",
    ])

    # Create ablation study for specific components
    ablation = AblationStudy(
        base_args=args,
        components=["drop_edge", "bold_noise"],
        output_dir="results/demo_ablations",
    )

    print("Will ablate:")
    for comp in ablation.components:
        print(f"  - {comp.name}: {comp.description}")

    # NOTE: Uncomment to actually run
    # ablation.run()
    # print(ablation.summary())
    # ablation.save_results()

    print("\n  (Run with: ablation.run() to execute)")


def example_3_cross_validation():
    """Example 3: K-fold cross-validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: K-Fold Cross-Validation")
    print("=" * 70)

    parser = build_parser()
    args = parser.parse_args([
        "--model_name", "graph_temporal",
        "--max_epochs", "1",
        "--batch_size", "4",
        "--accelerator", "cpu",
    ])

    print("Will run 5-fold stratified cross-validation")
    print("Expected output: cv_summary.json with mean/std metrics")

    # NOTE: Uncomment to actually run
    # cv_results = kfold_cross_validate(args, n_splits=5, output_dir="results/demo_cv")
    # print(cv_results.mean_metrics())

    print("\n  (Run with: kfold_cross_validate(...) to execute)")


def example_4_extended_metrics():
    """Example 4: Compute extended evaluation metrics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Extended Evaluation Metrics")
    print("=" * 70)

    # Simulate predictions and labels
    np.random.seed(42)
    n_test = 100

    # Create synthetic predictions and labels
    probs = np.random.uniform(0, 1, n_test)
    labels = np.random.randint(0, 2, n_test)

    # Compute metrics
    metrics = compute_metrics(probs, labels)

    print("\nClassification Metrics:")
    print(f"  Accuracy:   {metrics.accuracy:.4f}")
    print(f"  Sensitivity: {metrics.sensitivity:.4f}  (ASD recall)")
    print(f"  Specificity: {metrics.specificity:.4f}  (TD recall)")
    print(f"  Precision:  {metrics.precision:.4f}")
    print(f"  F1 Score:   {metrics.f1:.4f}")
    print(f"  AUC:        {metrics.auc:.4f}")
    print(f"  MCC:        {metrics.mcc:.4f}")
    print(f"  Kappa:      {metrics.kappa:.4f}")

    # Compute ROC
    roc = compute_roc_curve(probs, labels)
    print(f"\nROC Curve AUC: {roc['auc']:.4f}")


def example_5_hpo_suggestion():
    """Example 5: HPO configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Hyperparameter Optimization Setup")
    print("=" * 70)

    # Create HPO configuration
    config = HPOConfig(
        study_name="demo_hpo",
        n_trials=5,  # Small for demo
        objective_metric="test_auc",
        seed=42,
    )

    print(f"HPO Configuration:")
    print(f"  Study name: {config.study_name}")
    print(f"  N trials: {config.n_trials}")
    print(f"  Objective: {config.objective_metric}")
    print(f"  Direction: {config.direction}")

    print(f"\nSearch Space:")
    print(f"  hidden_dim ∈ {{32, 64, 128, 256}}")
    print(f"  dropout ∈ [0.0, 0.5]")
    print(f"  lr ∈ [1e-5, 1e-2]")
    print(f"  batch_size ∈ {{8, 16, 32, 64}}")
    print(f"  drop_edge_p ∈ [0.0, 0.3]")
    print(f"  bold_noise_std ∈ [0.0, 0.05]")

    print(f"\nTo run HPO:")
    print(f"  python -m brain_gcn.hpo_cli --hpo_n_trials 20 --hpo_objective test_auc")

    # NOTE: Uncomment to actually run
    # study = HPOStudy(config)
    # study.create_study()
    # study.optimize(base_args)
    # print(f"Best: {study.best_value():.4f}")
    # study.save_summary("results/hpo_summary.json")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Brain-Connectivity-GCN: Experiment Management Quick Start")
    print("=" * 70)

    example_1_basic_training()
    example_2_ablation_study()
    example_3_cross_validation()
    example_4_extended_metrics()
    example_5_hpo_suggestion()

    print("\n" + "=" * 70)
    print("Examples completed! See EXPERIMENTS.md for full documentation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
