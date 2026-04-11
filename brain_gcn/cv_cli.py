"""
K-fold cross-validation entry point.

Usage:
    python -m brain_gcn.cv_cli --n_splits 5 --cv_output_dir results/cv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from brain_gcn.main import build_parser
from brain_gcn.utils.cross_validation import kfold_cross_validate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def add_cv_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CV-specific arguments."""
    parser.add_argument(
        "--cv_n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--cv_output_dir",
        type=str,
        default="results/cv",
        help="Output directory for CV results.",
    )
    return parser


def main():
    parser = build_parser()
    parser = add_cv_arguments(parser)
    args = parser.parse_args()

    log.info(f"Starting {args.cv_n_splits}-fold cross-validation")
    log.info(f"Model: {args.model_name}")
    log.info(f"Output: {args.cv_output_dir}")

    # Run cross-validation
    cv_results = kfold_cross_validate(
        args,
        n_splits=args.cv_n_splits,
        output_dir=args.cv_output_dir,
    )

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("CROSS-VALIDATION COMPLETE")
    log.info("=" * 70)

    summary = cv_results.mean_metrics()
    for key, value in sorted(summary.items()):
        if isinstance(value, float):
            log.info(f"{key}: {value:.4f}")

    # Save summary
    summary_file = Path(args.cv_output_dir) / "cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(cv_results.to_dict(), f, indent=2)

    log.info(f"\nResults saved to {summary_file}")


if __name__ == "__main__":
    main()
