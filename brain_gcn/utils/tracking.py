"""
Experiment tracking and logging infrastructure.

Tracks:
- Run metadata (config, environment, hardware)
- Training/validation/test metrics
- Checkpoint locations
- Results summaries
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import platform
import torch

log = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""

    run_id: str
    timestamp: str
    model_name: str
    dataset: str = "ABIDE"
    split_strategy: str = "site_holdout"
    notes: str = ""

    # Environment
    python_version: str = ""
    pytorch_version: str = ""
    device: str = ""
    num_gpus: int = 0

    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Results
    test_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_args(
        cls,
        run_id: str,
        args,
        notes: str = "",
    ) -> ExperimentMetadata:
        """Create metadata from training arguments.

        Parameters
        ----------
        run_id : str
            Unique run identifier.
        args : argparse.Namespace
            Training arguments.
        notes : str, optional
            Additional notes.

        Returns
        -------
        ExperimentMetadata
            Metadata object.
        """
        hyperparams = {
            "hidden_dim": getattr(args, "hidden_dim", None),
            "dropout": getattr(args, "dropout", None),
            "lr": getattr(args, "lr", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "batch_size": getattr(args, "batch_size", None),
            "max_epochs": getattr(args, "max_epochs", None),
            "drop_edge_p": getattr(args, "drop_edge_p", None),
            "bold_noise_std": getattr(args, "bold_noise_std", None),
        }

        return cls(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model_name=getattr(args, "model_name", "unknown"),
            split_strategy=getattr(args, "split_strategy", "site_holdout"),
            notes=notes,
            python_version=platform.python_version(),
            pytorch_version=torch.__version__,
            device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            num_gpus=torch.cuda.device_count(),
            hyperparameters=hyperparams,
        )


class ExperimentTracker:
    """Tracks and logs experiment runs."""

    def __init__(self, output_dir: str | Path = "experiments"):
        """Initialize tracker.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save experiment logs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_list: list[ExperimentMetadata] = []

    def add_run(
        self,
        metadata: ExperimentMetadata,
    ) -> None:
        """Record a completed run.

        Parameters
        ----------
        metadata : ExperimentMetadata
            Run metadata.
        """
        self.metadata_list.append(metadata)
        self._save_run(metadata)

    def _save_run(self, metadata: ExperimentMetadata) -> None:
        """Save individual run to JSON."""
        run_dir = self.output_dir / metadata.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta_file = run_dir / "metadata.json"
        with open(meta_file, "w") as f:
            f.write(metadata.to_json())

        log.info(f"Experiment metadata saved to {meta_file}")

    def save_summary(self) -> None:
        """Save summary of all runs."""
        summary_file = self.output_dir / "summary.json"

        summary = {
            "total_runs": len(self.metadata_list),
            "runs": [m.to_dict() for m in self.metadata_list],
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        log.info(f"Experiment summary saved to {summary_file}")

    def load_summary(self) -> dict:
        """Load summary from disk."""
        summary_file = self.output_dir / "summary.json"
        if not summary_file.exists():
            return {"total_runs": 0, "runs": []}

        with open(summary_file) as f:
            return json.load(f)


class RunLogger:
    """Context manager for logging a single run."""

    def __init__(
        self,
        run_id: str,
        args,
        tracker: ExperimentTracker,
        notes: str = "",
    ):
        """Initialize run logger.

        Parameters
        ----------
        run_id : str
            Unique run ID.
        args : argparse.Namespace
            Training arguments.
        tracker : ExperimentTracker
            Parent tracker.
        notes : str, optional
            Notes about the run.
        """
        self.run_id = run_id
        self.args = args
        self.tracker = tracker
        self.notes = notes
        self.metadata = ExperimentMetadata.from_args(run_id, args, notes)

    def __enter__(self):
        """Enter context."""
        log.info(f"Starting run: {self.run_id}")
        return self.metadata

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log results."""
        if exc_type is not None:
            log.error(f"Run {self.run_id} failed: {exc_val}")
            return

        self.tracker.add_run(self.metadata)
        log.info(f"Run {self.run_id} completed and logged")

    def update_metrics(self, metrics: dict) -> None:
        """Update test metrics."""
        self.metadata.test_metrics.update(metrics)

    def set_checkpoint_path(self, path: str | Path) -> None:
        """Record checkpoint location."""
        self.metadata.checkpoint_path = str(path)
