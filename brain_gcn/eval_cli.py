"""
Evaluation entry point for extended metrics analysis.

Computes extended evaluation metrics, ROC curves, and statistical tests.

Usage:
    python -m brain_gcn.eval_cli --checkpoint <path> --test_metrics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc

from brain_gcn.main import build_datamodule
from brain_gcn.tasks import ClassificationTask
from brain_gcn.utils.evaluation import (
    compute_metrics,
    compute_roc_curve,
    compute_pr_curve,
    compute_confusion_matrix,
    StatisticalTester,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_eval_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add evaluation-specific arguments."""
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="results/evaluation",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--eval_plot_roc",
        action="store_true",
        help="Save ROC curve plot.",
    )
    parser.add_argument(
        "--eval_plot_pr",
        action="store_true",
        help="Save Precision-Recall curve plot.",
    )
    parser.add_argument(
        "--eval_bootstrap_ci",
        action="store_true",
        help="Compute bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--eval_ci_n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples.",
    )
    return parser


def load_checkpoint(
    ckpt_path: str | Path,
    device: str = "cpu",
) -> ClassificationTask:
    """Load trained model from checkpoint."""
    return ClassificationTask.load_from_checkpoint(ckpt_path, map_location=device)


def get_predictions(
    model: ClassificationTask,
    dm,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions on test set."""
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for bold_windows, adj, labels in dm.test_dataloader():
            logits = model(bold_windows.to(device), adj.to(device))
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_roc(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot and save ROC curve."""
    roc_data = compute_roc_curve(probs, labels)
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc_score = roc_data["auc"]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    log.info(f"ROC curve saved to {output_path}")


def plot_pr(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot and save Precision-Recall curve."""
    pr_data = compute_pr_curve(probs, labels)
    precision = pr_data["precision"]
    recall = pr_data["recall"]
    ap = pr_data["ap"]

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR (AP={ap:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    log.info(f"PR curve saved to {output_path}")


def main():
    from brain_gcn.main import build_parser

    parser = build_parser()
    parser = add_eval_arguments(parser)
    args = parser.parse_args()

    output_dir = Path(args.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    log.info(f"Loading checkpoint: {args.eval_checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.eval_checkpoint, device=device)

    log.info("Building datamodule")
    dm = build_datamodule(args)
    dm.prepare_data()
    dm.setup()

    # Get predictions
    log.info("Generating predictions on test set")
    probs, labels = get_predictions(model, dm, device=device)

    # Compute metrics
    log.info("Computing metrics")
    metrics = compute_metrics(probs, labels)
    cm = compute_confusion_matrix(probs, labels)

    # Print metrics
    log.info("\n" + "=" * 70)
    log.info("CLASSIFICATION METRICS")
    log.info("=" * 70)
    for key, value in metrics.to_dict().items():
        log.info(f"{key:20s}: {value:.4f}")

    log.info("\nConfusion Matrix:")
    log.info(f"  TP={cm.true_positives}, FP={cm.false_positives}")
    log.info(f"  FN={cm.false_negatives}, TN={cm.true_negatives}")

    # Compute confidence intervals if requested
    if args.eval_bootstrap_ci:
        log.info(f"\nComputing {args.eval_ci_n_bootstrap} bootstrap samples")
        ci_auc = StatisticalTester.bootstrap_ci(
            lambda p, l: compute_metrics(p, l).auc,
            probs,
            labels,
            n_bootstrap=args.eval_ci_n_bootstrap,
        )
        log.info(f"AUC 95% CI: [{ci_auc[0]:.4f}, {ci_auc[2]:.4f}]")

    # Save results
    results = {
        "metrics": metrics.to_dict(),
        "confusion_matrix": cm.to_dict(),
    }

    results_file = output_dir / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved to {results_file}")

    # Plot ROC and PR curves if requested
    if args.eval_plot_roc:
        roc_path = output_dir / "roc_curve.png"
        plot_roc(probs, labels, roc_path)

    if args.eval_plot_pr:
        pr_path = output_dir / "pr_curve.png"
        plot_pr(probs, labels, pr_path)


if __name__ == "__main__":
    main()
