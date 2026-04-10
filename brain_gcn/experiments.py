"""
Multi-model comparison runner.

v2 changes:
  - Captures test_sens, test_spec, and ensemble metrics in results CSV
  - Passes dynamic_graph_temporal flag through correctly
  - Uses site_holdout as default (inherited from updated main.py defaults)
"""

from __future__ import annotations

import argparse
import csv
import logging
from copy import deepcopy
from pathlib import Path

import torch

from brain_gcn.main import build_parser, train_from_args, validate_args

log = logging.getLogger(__name__)


DEFAULT_MODELS = ("fc_mlp", "gru", "gcn", "graph_temporal")


def metric_value(value) -> float | int | str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        # Multi-element tensor: flatten to scalar_mean or scalar_max
        scalar_mean = float(value.detach().cpu().mean())
        log.warning(
            f"Multi-element metric tensor with shape {value.shape} — "
            f"flattening to scalar_mean={scalar_mean:.4f}. "
            "Consider reducing to single-value metrics in training_step."
        )
        return scalar_mean
    if isinstance(value, (float, int, str)):
        return value
    return str(value)


def build_experiment_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Run Brain-Connectivity-GCN model comparisons"
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["fc_mlp", "gru", "gcn", "graph_temporal"],
        default=list(DEFAULT_MODELS),
        help="Model modes to run in order.",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="results/experiment_summary.csv",
    )
    parser.add_argument(
        "--dynamic_graph_temporal",
        action="store_true",
        help="Run graph_temporal with per-window adjacency sequences.",
    )
    parser.set_defaults(test=True)
    return parser


def args_for_model(base_args: argparse.Namespace, model_name: str) -> argparse.Namespace:
    args = deepcopy(base_args)
    args.model_name = model_name
    args.prepare_data = False

    if model_name == "fc_mlp":
        args.use_population_adj = False
        args.use_dynamic_adj_sequence = False
        args.use_dynamic_adj = False
    elif model_name == "graph_temporal" and args.dynamic_graph_temporal:
        args.use_population_adj = False
        args.use_dynamic_adj_sequence = True
        args.use_dynamic_adj = False
    else:
        args.use_dynamic_adj_sequence = False

    validate_args(args)
    return args


def summarize_run(model_name: str, trainer) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"model_name": model_name}
    for key, value in sorted(trainer.callback_metrics.items()):
        if key.startswith(("train_", "val_", "test_")):
            row[key] = metric_value(value)
    return row


def write_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    # model_name first, then alphabetical
    fieldnames = ["model_name"] + [k for k in fieldnames if k != "model_name"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_experiment_parser()
    args = parser.parse_args()

    # prepare and setup once (before the model loop)
    # Call setup() before preprocess_all so train_subjects reflects the actual split
    from brain_gcn.main import build_datamodule
    prep_args = deepcopy(args)
    prep_args.prepare_data = True
    dm = build_datamodule(prep_args)
    dm.prepare_data()
    dm.setup()  # Call setup here to establish actual train/val/test boundary

    rows = []
    for model_name in args.models:
        run_args = args_for_model(args, model_name)
        trainer, _, _ = train_from_args(run_args)
        rows.append(summarize_run(model_name, trainer))
        write_results(Path(args.results_csv), rows)
        print(f"[{model_name}] done — partial results written to {args.results_csv}")

    print(f"\nWrote {len(rows)} rows to {args.results_csv}")


if __name__ == "__main__":
    main()
