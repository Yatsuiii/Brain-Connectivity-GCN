"""
Hyperparameter optimization using Optuna.

Provides automated search over model, training, and data hyperparameters.
Integrates with the existing training pipeline via argparse.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial

from brain_gcn.main import train_from_args, validate_args

log = logging.getLogger(__name__)


class HPOConfig:
    """Hyperparameter optimization configuration."""

    def __init__(
        self,
        study_name: str = "brain_gcn_hpo",
        n_trials: int = 20,
        timeout: int | None = None,
        direction: str = "maximize",
        objective_metric: str = "test_auc",
        storage: str | None = None,
        seed: int = 42,
    ):
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.objective_metric = objective_metric
        self.storage = storage
        self.seed = seed


class HPOSearchSpace:
    """Define hyperparameter search space for Optuna."""

    @staticmethod
    def suggest_params(trial: Trial, base_args: argparse.Namespace) -> argparse.Namespace:
        """Suggest hyperparameters for a single trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            Current trial object.
        base_args : argparse.Namespace
            Base arguments; suggested values override these.

        Returns
        -------
        argparse.Namespace
            Arguments with suggested hyperparameters.
        """
        args = argparse.Namespace(**vars(base_args))

        # Model architecture
        args.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [32, 64, 128, 256]
        )
        args.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        # Training
        args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
        args.batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64]
        )

        # DropEdge regularization
        args.drop_edge_p = trial.suggest_float("drop_edge_p", 0.0, 0.3, step=0.1)

        # BOLD noise augmentation
        args.bold_noise_std = trial.suggest_float(
            "bold_noise_std", 0.0, 0.05, step=0.01
        )

        # Cosine annealing
        args.cosine_t0 = trial.suggest_categorical(
            "cosine_t0", [30, 50, 100]
        )
        args.cosine_t_mult = trial.suggest_categorical(
            "cosine_t_mult", [1, 2, 3]
        )
        args.cosine_eta_min = trial.suggest_loguniform(
            "cosine_eta_min", 1e-6, 1e-4
        )

        return args


def objective(
    trial: Trial,
    base_args: argparse.Namespace,
    hpo_config: HPOConfig,
) -> float:
    """Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current trial.
    base_args : argparse.Namespace
        Base arguments template.
    hpo_config : HPOConfig
        HPO configuration.

    Returns
    -------
    float
        Objective value (test set metric).
    """
    try:
        # Suggest hyperparameters
        args = HPOSearchSpace.suggest_params(trial, base_args)
        validate_args(args)

        # Train model
        trainer, _, _ = train_from_args(args)

        # Extract objective metric
        metric_value = trainer.callback_metrics.get(
            hpo_config.objective_metric,
            None
        )
        if metric_value is None:
            log.warning(
                f"Metric {hpo_config.objective_metric} not found. "
                "Available: %s", list(trainer.callback_metrics.keys())
            )
            return float("-inf")

        return float(metric_value.detach().cpu())

    except Exception as e:
        log.error(f"Trial failed: {e}")
        return float("-inf")


class HPOStudy:
    """Wrapper for Optuna study with convenience methods."""

    def __init__(self, config: HPOConfig):
        self.config = config
        self.study: optuna.Study | None = None

    def create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        sampler = TPESampler(seed=self.config.seed)
        pruner = MedianPruner()

        storage_url = None
        if self.config.storage:
            storage_url = f"sqlite:///{self.config.storage}"

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True,
        )
        return self.study

    def optimize(
        self,
        base_args: argparse.Namespace,
    ) -> optuna.Study:
        """Run hyperparameter optimization.

        Parameters
        ----------
        base_args : argparse.Namespace
            Base arguments template.

        Returns
        -------
        optuna.Study
            Completed study object.
        """
        if self.study is None:
            self.create_study()

        self.study.optimize(
            lambda trial: objective(trial, base_args, self.config),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )
        return self.study

    def best_params(self) -> dict[str, Any]:
        """Get best hyperparameters found."""
        if self.study is None:
            raise RuntimeError("Study not created. Call optimize() first.")
        return self.study.best_params

    def best_value(self) -> float:
        """Get best objective value."""
        if self.study is None:
            raise RuntimeError("Study not created. Call optimize() first.")
        return self.study.best_value

    def save_summary(self, output_path: str | Path) -> None:
        """Save HPO summary to JSON."""
        if self.study is None:
            raise RuntimeError("Study not created. Call optimize() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "study_name": self.config.study_name,
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "direction": self.config.direction,
            "objective_metric": self.config.objective_metric,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        log.info(f"HPO summary saved to {output_path}")


def hpo_from_args(args: argparse.Namespace) -> HPOStudy:
    """Create HPO study from command-line arguments."""
    hpo_config = HPOConfig(
        study_name=args.hpo_study_name,
        n_trials=args.hpo_n_trials,
        timeout=args.hpo_timeout,
        objective_metric=args.hpo_objective,
        storage=args.hpo_storage,
        seed=args.seed,
    )
    return HPOStudy(hpo_config)


def add_hpo_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add HPO-specific arguments to parser."""
    parser.add_argument(
        "--hpo_study_name",
        type=str,
        default="brain_gcn_hpo",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--hpo_n_trials",
        type=int,
        default=20,
        help="Number of trials.",
    )
    parser.add_argument(
        "--hpo_timeout",
        type=int,
        default=None,
        help="Timeout in seconds.",
    )
    parser.add_argument(
        "--hpo_objective",
        type=str,
        default="test_auc",
        help="Metric to optimize (e.g., test_auc, test_acc).",
    )
    parser.add_argument(
        "--hpo_storage",
        type=str,
        default="hpo_studies.db",
        help="SQLite storage path for persistent studies.",
    )
    return parser
