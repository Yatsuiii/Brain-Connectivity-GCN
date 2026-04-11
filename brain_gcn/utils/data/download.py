"""
Download ABIDE I preprocessed ROI time series via nilearn.

Uses the Preprocessed Connectomes Project (PCP) pipeline:
  - pipeline:     cpac  (most widely used in ABIDE literature)
  - band-pass:    on
  - GSR:          on   (global signal regression)
  - atlas:        cc200 (200 ROIs, Craddock 2012) — pre-extracted, no NIfTI needed

In nilearn >= 0.10, rois_cc200 returns numpy arrays directly (not file paths).
Total download size for full ABIDE I (~884 usable subjects): ~80 MB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from nilearn import datasets

log = logging.getLogger(__name__)

SUBJECT_ID_COL = "SUB_ID"
LABEL_COL = "DX_GROUP"   # 1 = ASD, 2 = Typical Control
SITE_COL = "SITE_ID"


def fetch_abide(
    data_dir: str | Path,
    n_subjects: int | None = None,
    pipeline: str = "cpac",
    band_pass_filtering: bool = True,
    global_signal_regression: bool = True,
    derivatives: list[str] | None = None,
) -> datasets.utils.Bunch:
    """
    Download ABIDE I preprocessed data via nilearn.

    Returns a nilearn Bunch with:
        .rois_cc200   list of (T, 200) numpy arrays — one per subject
        .phenotypic   pandas DataFrame with subject metadata
    """
    if derivatives is None:
        derivatives = ["rois_cc200"]

    log.info("Fetching ABIDE I (pipeline=%s, n_subjects=%s) ...", pipeline, n_subjects)
    dataset = datasets.fetch_abide_pcp(
        data_dir=str(data_dir),
        n_subjects=n_subjects,
        pipeline=pipeline,
        band_pass_filtering=band_pass_filtering,
        global_signal_regression=global_signal_regression,
        derivatives=derivatives,
        verbose=0,
    )
    log.info("Fetched %d subjects.", len(dataset.rois_cc200))
    return dataset


def get_label(phenotypic_row) -> int:
    """DX_GROUP: 1 = ASD, 2 = Typical Control  →  ASD=1, TC=0"""
    dx = int(phenotypic_row[LABEL_COL])
    assert dx in (1, 2), f"Unexpected DX_GROUP value: {dx}. Must be 1 (ASD) or 2 (TC)."
    return 1 if dx == 1 else 0


def extract_subjects(
    dataset: datasets.utils.Bunch,
    min_timepoints: int = 100,
) -> list[dict]:
    """
    Validate and pair each subject's BOLD array with its label and metadata.

    Returns list of dicts with keys:
        subject_id, site, label, bold (np.ndarray T×N)
    """
    pheno = dataset.phenotypic
    arrays = dataset.rois_cc200

    subjects = []
    dropped = 0

    for i, bold in enumerate(arrays):
        bold = np.array(bold, dtype=np.float32)

        if bold.ndim != 2:
            log.warning("Subject %d: unexpected shape %s — skipping.", i, bold.shape)
            dropped += 1
            continue

        if not np.isfinite(bold).all():
            log.warning("Subject %d: NaN/Inf values — skipping.", i)
            dropped += 1
            continue

        if bold.shape[0] < min_timepoints:
            log.debug(
                "Subject %d: only %d TRs (min=%d) — skipping.",
                i, bold.shape[0], min_timepoints,
            )
            dropped += 1
            continue

        row = pheno.iloc[i]
        subjects.append({
            "subject_id": str(row[SUBJECT_ID_COL]),
            "site": str(row[SITE_COL]),
            "label": get_label(row),
            "bold": bold,                   # (T, N) float32
            "n_timepoints": bold.shape[0],
        })

    log.info("Kept %d subjects, dropped %d.", len(subjects), dropped)
    return subjects
