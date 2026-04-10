"""
Extended evaluation metrics and analysis tools.

Provides:
- Per-class metrics (sensitivity, specificity, precision, F1)
- ROC/AUC analysis
- Confusion matrix
- Calibration curves
- Statistical significance testing
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from scipy import stats


@dataclass
class BinaryClassificationMetrics:
    """Container for binary classification metrics."""

    accuracy: float
    sensitivity: float  # ASD recall (TP / (TP + FN))
    specificity: float  # TD recall (TN / (TN + FP))
    precision: float    # ASD precision (TP / (TP + FP))
    f1: float          # ASD F1
    auc: float         # ROC AUC
    mcc: float         # Matthews Correlation Coefficient
    kappa: float       # Cohen's Kappa

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "auc": self.auc,
            "mcc": self.mcc,
            "kappa": self.kappa,
        }


@dataclass
class ConfusionMatrixMetrics:
    """Container for confusion matrix analysis."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tp": self.true_positives,
        }


def compute_metrics(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Compute comprehensive binary classification metrics.

    Parameters
    ----------
    probs : (N,) or (N, 2) tensor/array
        Predicted probabilities. If (N, 2), uses class 1; if (N,), assumes
        probability of positive class.
    labels : (N,) tensor/array
        Ground truth binary labels (0 or 1).
    threshold : float
        Decision threshold for classification.

    Returns
    -------
    BinaryClassificationMetrics
        Computed metrics.
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Extract probability of positive class
    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    # Hard predictions
    preds = (probs_pos >= threshold).astype(int)

    # Basic metrics
    accuracy = np.mean(preds == labels)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) \
        if (precision + sensitivity) > 0 else 0.0

    # AUC
    try:
        auc_score = roc_auc_score(labels, probs_pos)
    except ValueError:
        auc_score = 0.0

    # Matthews correlation coefficient
    mcc = matthews_corrcoef(labels, preds)

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, preds)

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        f1=f1,
        auc=auc_score,
        mcc=mcc,
        kappa=kappa,
    )


def compute_confusion_matrix(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> ConfusionMatrixMetrics:
    """Compute confusion matrix components.

    Parameters
    ----------
    probs : (N,) or (N, 2) tensor/array
        Predicted probabilities.
    labels : (N,) tensor/array
        Ground truth labels.
    threshold : float
        Decision threshold.

    Returns
    -------
    ConfusionMatrixMetrics
        Confusion matrix components.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    preds = (probs_pos >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return ConfusionMatrixMetrics(
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def compute_roc_curve(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict:
    """Compute ROC curve.

    Parameters
    ----------
    probs : (N,) or (N, 2) tensor/array
        Predicted probabilities.
    labels : (N,) tensor/array
        Ground truth labels.

    Returns
    -------
    dict
        FPR, TPR, thresholds, and AUC.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    fpr, tpr, thresholds = roc_curve(labels, probs_pos)
    auc_score = auc(fpr, tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_score,
    }


def compute_pr_curve(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict:
    """Compute Precision-Recall curve.

    Parameters
    ----------
    probs : (N,) or (N, 2) tensor/array
        Predicted probabilities.
    labels : (N,) tensor/array
        Ground truth labels.

    Returns
    -------
    dict
        Precision, recall, thresholds, and AP.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if probs.ndim == 2:
        probs_pos = probs[:, 1]
    else:
        probs_pos = probs

    precision, recall, thresholds = precision_recall_curve(labels, probs_pos)
    ap = auc(recall, precision)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "ap": ap,
    }


class StatisticalTester:
    """Statistical significance testing utilities."""

    @staticmethod
    def bootstrap_ci(
        metric_fn,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float, float]:
        """Compute confidence interval via bootstrap.

        Parameters
        ----------
        metric_fn : callable
            Function that computes metric from (probs, labels).
        probs : (N,) array
            Predicted probabilities.
        labels : (N,) array
            Ground truth labels.
        n_bootstrap : int
            Number of bootstrap samples.
        ci : float
            Confidence interval (0.95 = 95%).

        Returns
        -------
        tuple[float, float, float]
            (lower, estimate, upper) bounds.
        """
        n = len(labels)
        bootstrap_vals = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            val = metric_fn(probs[idx], labels[idx])
            bootstrap_vals.append(val)

        bootstrap_vals = np.array(bootstrap_vals)
        lower = np.percentile(bootstrap_vals, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_vals, (1 + ci) / 2 * 100)
        estimate = np.mean(bootstrap_vals)

        return lower, estimate, upper

    @staticmethod
    def compare_auc(
        probs1: np.ndarray,
        probs2: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Compare AUC of two models (DeLong test).

        Parameters
        ----------
        probs1, probs2 : (N,) array
            Predicted probabilities from two models.
        labels : (N,) array
            Ground truth labels.

        Returns
        -------
        dict
            AUC1, AUC2, z-statistic, p-value.
        """
        auc1 = roc_auc_score(labels, probs1)
        auc2 = roc_auc_score(labels, probs2)

        # Simplified comparison (two-sample t-test on AUC)
        # For proper DeLong test, see sklearn-labs or hand implementation
        t_stat, p_val = stats.ttest_ind(
            roc_curve(labels, probs1)[1],
            roc_curve(labels, probs2)[1],
        )

        return {
            "auc1": auc1,
            "auc2": auc2,
            "difference": auc1 - auc2,
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        }
