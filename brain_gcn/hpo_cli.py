"""
Hyperparameter search entry point.

Usage:
    python -m brain_gcn.hpo_cli --hpo_n_trials 20 --hpo_objective test_auc
"""

from __future__ import annotations

import argparse
import logging

from brain_gcn.hpo import add_hpo_arguments, hpo_from_args
from brain_gcn.main import build_parser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = build_parser()
    parser = add_hpo_arguments(parser)
    args = parser.parse_args()

    log.info(f"Starting HPO study: {args.hpo_study_name}")
    log.info(f"Objective: {args.hpo_objective}")
    log.info(f"N trials: {args.hpo_n_trials}")

    # Create and run HPO study
    hpo_study = hpo_from_args(args)
    study = hpo_study.optimize(args)

    # Print results
    log.info("\n" + "=" * 70)
    log.info("OPTIMIZATION COMPLETE")
    log.info("=" * 70)
    log.info(f"Best value: {study.best_value:.4f}")
    log.info("\nBest hyperparameters:")
    for key, value in sorted(study.best_params.items()):
        log.info(f"  {key}: {value}")

    # Save summary
    hpo_study.save_summary("results/hpo_summary.json")


if __name__ == "__main__":
    main()
