"""
Ablation study framework.

Systematically removes or disables components to measure their contribution.

Examples:
  - Disable DropEdge (set drop_edge_p=0)
  - Disable BOLD augmentation (set bold_noise_std=0)
  - Use GCN baseline vs full graph-temporal
  - Population adj vs per-subject adjacency
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytorch_lightning as pl
import torch

from brain_gcn.main import train_from_args, validate_args

log = logging.getLogger(__name__)


@dataclass
class AblationComponent:
    """Single component to ablate."""

    name: str
    description: str
    modify_fn: Callable[[argparse.Namespace], argparse.Namespace]
    enabled: bool = True


class AblationStudy:
    """Framework for systematic ablation studies."""

    # Predefined components
    COMPONENTS = {
        "drop_edge": AblationComponent(
            name="drop_edge",
            description="DropEdge regularization in graph convolution",
            modify_fn=lambda args: (setattr(args, "drop_edge_p", 0.0), args)[1],
        ),
        "bold_noise": AblationComponent(
            name="bold_noise",
            description="BOLD signal augmentation during training",
            modify_fn=lambda args: (setattr(args, "bold_noise_std", 0.0), args)[1],
        ),
        "graph": AblationComponent(
            name="graph",
            description="Graph structure (use GRU-only baseline)",
            modify_fn=lambda args: (setattr(args, "model_name", "gru"), args)[1],
        ),
        "population_adj": AblationComponent(
            name="population_adj",
            description="Population adjacency matrix",
            modify_fn=lambda args: (setattr(args, "use_population_adj", False), args)[1],
        ),
        "layer_norm": AblationComponent(
            name="layer_norm",
            description="Layer normalization in graph convolutions",
            modify_fn=lambda args: (setattr(args, "use_layer_norm", False), args)[1],
        ),
    }

    def __init__(
        self,
        base_args: argparse.Namespace,
        components: list[str] | None = None,
        output_dir: str | Path | None = None,
    ):
        """Initialize ablation study.

        Parameters
        ----------
        base_args : argparse.Namespace
            Base training arguments (full model).
        components : list[str], optional
            List of component names to ablate. If None, ablates all.
        output_dir : str or Path, optional
            Directory to save results.
        """
        self.base_args = deepcopy(base_args)
        self.output_dir = Path(output_dir) if output_dir else Path("ablations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if components is None:
            self.component_names = list(self.COMPONENTS.keys())
        else:
            self.component_names = components

        self.components = [
            self.COMPONENTS[name] for name in self.component_names
            if name in self.COMPONENTS
        ]

        self.results: dict[str, dict] = {}

    def run(self) -> dict[str, dict]:
        """Run full ablation study.

        Returns
        -------
        dict[str, dict]
            Results keyed by component name.
        """
        # Train full model first
        log.info("Training full model (baseline)")
        pl.seed_everything(self.base_args.seed, workers=True)
        try:
            trainer, _, _ = train_from_args(self.base_args)
            baseline_metrics = {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in trainer.callback_metrics.items()
                if key.startswith(("test_",))
            }
        except Exception as e:
            log.error(f"Baseline training failed: {e}")
            baseline_metrics = {}

        self.results["baseline"] = baseline_metrics

        # Ablate each component
        for component in self.components:
            log.info(f"Ablating: {component.name} ({component.description})")

            ablated_args = deepcopy(self.base_args)
            ablated_args = component.modify_fn(ablated_args)

            try:
                validate_args(ablated_args)
            except ValueError as e:
                log.warning(f"Ablation {component.name} skipped: {e}")
                continue

            pl.seed_everything(self.base_args.seed, workers=True)
            try:
                trainer, _, _ = train_from_args(ablated_args)
                ablated_metrics = {
                    key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in trainer.callback_metrics.items()
                    if key.startswith(("test_",))
                }
            except Exception as e:
                log.error(f"Ablation {component.name} failed: {e}")
                ablated_metrics = {}

            self.results[component.name] = ablated_metrics

        # Compute deltas
        self._compute_deltas(baseline_metrics)

        return self.results

    def _compute_deltas(self, baseline: dict) -> None:
        """Compute metric changes from baseline."""
        deltas = {}

        for component_name, ablated_metrics in self.results.items():
            if component_name == "baseline":
                deltas[component_name] = {}
                continue

            delta = {}
            for key, ablated_val in ablated_metrics.items():
                baseline_val = baseline.get(key, None)
                if baseline_val is not None and isinstance(ablated_val, (int, float)):
                    delta[key] = ablated_val - baseline_val
                else:
                    delta[key] = None

            deltas[component_name] = delta

        self.deltas = deltas

    def save_results(self) -> None:
        """Save results to JSON."""
        results_file = self.output_dir / "ablation_results.json"

        # Convert torch tensors to serializable format
        serializable = {}
        for key, metrics in self.results.items():
            serializable[key] = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in metrics.items()
            }

        deltas_serializable = {}
        for key, deltas in self.deltas.items():
            deltas_serializable[key] = {
                k: float(v) if v is None or isinstance(v, (int, float)) else str(v)
                for k, v in deltas.items()
            }

        output = {
            "results": serializable,
            "deltas": deltas_serializable,
            "components": [c.name for c in self.components],
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"Ablation results saved to {results_file}")

    def summary(self) -> str:
        """Pretty-print summary."""
        lines = ["=" * 70]
        lines.append("ABLATION STUDY SUMMARY")
        lines.append("=" * 70)

        # Baseline
        if "baseline" in self.results:
            lines.append("\nBaseline (Full Model):")
            for key, val in sorted(self.results["baseline"].items()):
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")

        # Ablations
        lines.append("\nAblation Impact (Δ from Baseline):")
        lines.append("-" * 70)

        for component_name in self.component_names:
            if component_name in self.deltas:
                delta = self.deltas[component_name]
                lines.append(f"\n{component_name}:")
                for key, val in sorted(delta.items()):
                    if isinstance(val, float):
                        sign = "+" if val >= 0 else "-"
                        lines.append(f"  {key}: {sign}{abs(val):.4f}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def add_ablation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ablation-specific arguments."""
    parser.add_argument(
        "--ablation_components",
        nargs="+",
        choices=list(AblationStudy.COMPONENTS.keys()),
        help="Components to ablate. If not specified, ablates all.",
    )
    parser.add_argument(
        "--ablation_output_dir",
        type=str,
        default="results/ablations",
        help="Output directory for ablation results.",
    )
    return parser
