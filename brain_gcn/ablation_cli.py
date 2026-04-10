"""
Ablation study entry point.

Usage:
    python -m brain_gcn.ablation_cli --ablation_components drop_edge bold_noise
"""

from __future__ import annotations

import argparse
import logging

from brain_gcn.ablation import AblationStudy, add_ablation_arguments
from brain_gcn.main import build_parser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = build_parser()
    parser = add_ablation_arguments(parser)
    args = parser.parse_args()

    components = args.ablation_components or list(AblationStudy.COMPONENTS.keys())

    log.info(f"Starting ablation study")
    log.info(f"Dataset split: {args.split_strategy}")
    log.info(f"Components to ablate: {', '.join(components)}")

    # Run ablation study
    ablation = AblationStudy(
        args,
        components=components,
        output_dir=args.ablation_output_dir,
    )
    ablation.run()

    # Print summary
    log.info(ablation.summary())

    # Save results
    ablation.save_results()


if __name__ == "__main__":
    main()
