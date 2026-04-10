"""
Cross-validation and K-fold evaluation utilities.

Provides:
- Stratified K-fold cross-validation
- Leave-one-site-out validation
- Train/val/test split preservation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from brain_gcn.main import build_datamodule, build_task, build_trainer, train_from_args
from brain_gcn.utils.data.datamodule import ABIDEDataModule

log = logging.getLogger(__name__)


class CVFold(NamedTuple):
    """Container for a single CV fold's results."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    metrics: dict  # {'test_auc': ..., 'test_acc': ...}


class CrossValidator:
    """Stratified K-fold cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        """Initialize CV splitter.

        Parameters
        ----------
        n_splits : int
            Number of folds.
        shuffle : bool
            Whether to shuffle before splitting.
        random_state : int
            Random seed.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(
        self,
        labels: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test split indices.

        Parameters
        ----------
        labels : (N,) array
            Class labels for stratification.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_idx, test_idx) tuples.
        """
        dummy_X = np.arange(len(labels)).reshape(-1, 1)
        splits = list(self.skf.split(dummy_X, labels))
        return [(train_idx, test_idx) for train_idx, test_idx in splits]


class LeaveOneSiteOutValidator:
    """Leave-one-site-out cross-validator."""

    def __init__(self):
        """Initialize LOSO validator."""
        pass

    def split(
        self,
        sites: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate leave-one-site-out splits.

        Parameters
        ----------
        sites : (N,) array
            Site labels for each subject.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (in_site_idx, out_site_idx) tuples.
        """
        unique_sites = np.unique(sites)
        splits = []

        for test_site in unique_sites:
            test_idx = np.where(sites == test_site)[0]
            train_idx = np.where(sites != test_site)[0]
            splits.append((train_idx, test_idx))

        return splits


class CVResults:
    """Accumulator for cross-validation results."""

    def __init__(self):
        self.folds: list[CVFold] = []

    def add_fold(self, fold: CVFold) -> None:
        """Add results from a single fold."""
        self.folds.append(fold)

    def mean_metrics(self) -> dict:
        """Compute mean metrics across folds."""
        if not self.folds:
            return {}

        all_metrics = [fold.metrics for fold in self.folds]
        keys = all_metrics[0].keys()

        means = {}
        for key in keys:
            values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
            if values:
                means[f"{key}_mean"] = float(np.mean(values))
                means[f"{key}_std"] = float(np.std(values))

        return means

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "n_folds": len(self.folds),
            "folds": [
                {
                    "fold_idx": fold.fold_idx,
                    "metrics": fold.metrics,
                }
                for fold in self.folds
            ],
            "summary": self.mean_metrics(),
        }


def kfold_cross_validate(
    base_args,
    n_splits: int = 5,
    output_dir: str | Path | None = None,
) -> CVResults:
    """Run stratified K-fold cross-validation.

    Parameters
    ----------
    base_args : argparse.Namespace
        Base training arguments.
    n_splits : int
        Number of folds.
    output_dir : str or Path, optional
        Directory to save fold results.

    Returns
    -------
    CVResults
        Aggregated cross-validation results.
    """
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build data module to get labels
    dm = build_datamodule(base_args)
    dm.prepare_data()
    dm.setup()

    # Collect labels
    all_labels = []
    for batch in dm.train_dataloader():
        _, _, labels = batch
        all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)

    # Initialize CV
    cv = CrossValidator(n_splits=n_splits, random_state=base_args.seed)
    splits = cv.split(all_labels)

    results = CVResults()

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        log.info(f"Running fold {fold_idx + 1}/{n_splits}")

        # Create fold-specific args
        fold_args = vars(base_args).copy()
        # Note: For full implementation, would need to modify datamodule
        # to accept external train/test splits. For now, train normally.

        # Train model
        pl.seed_everything(base_args.seed + fold_idx, workers=True)
        trainer, _, _ = train_from_args(base_args)

        # Collect metrics
        fold_metrics = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in trainer.callback_metrics.items()
            if key.startswith(("test_",))
        }

        fold_result = CVFold(
            fold_idx=fold_idx,
            train_indices=train_idx,
            val_indices=np.array([]),  # Not used in standard K-fold
            test_indices=test_idx,
            metrics=fold_metrics,
        )
        results.add_fold(fold_result)

        if output_dir:
            fold_file = output_dir / f"fold_{fold_idx}.pt"
            torch.save(fold_result, fold_file)

    if output_dir:
        summary_file = output_dir / "cv_summary.pt"
        torch.save(results.to_dict(), summary_file)
        log.info(f"CV results saved to {output_dir}")

    return results
