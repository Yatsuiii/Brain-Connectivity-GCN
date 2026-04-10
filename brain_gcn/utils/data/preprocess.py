"""
Preprocess ABIDE subjects into cached .npz files.

Each .npz contains:
    bold        (T, N)      — z-scored BOLD time series
    mean_fc     (N, N)      — full-scan Pearson FC
    bold_windows (W, N)      — mean BOLD per window (node features)
    fc_windows   (W, N, N)   — per-window Pearson FC (dynamic adjacency)
    label       scalar int  — 0 = TC, 1 = ASD
    subject_id  str
    site        str

Run once via ABIDEDataModule.prepare_data(); subsequent runs load from cache.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .functional_connectivity import compute_fc, sliding_fc_windows

log = logging.getLogger(__name__)


def zscore(bold: np.ndarray) -> np.ndarray:
    """Z-score each ROI time series independently."""
    mean = bold.mean(axis=0, keepdims=True)
    std = bold.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((bold - mean) / std).astype(np.float32)


def preprocess_subject(
    subject: dict,
    processed_dir: Path,
    window_len: int = 50,
    step: int = 5,
    overwrite: bool = False,
) -> Path | None:
    """
    Process one subject dict (from download.extract_subjects):
        z-score BOLD → compute FC + sliding windows → save .npz

    Returns Path to saved .npz, or None if processing failed.
    """
    out_path = processed_dir / f"{subject['subject_id']}.npz"

    if out_path.exists() and not overwrite:
        return out_path

    bold = subject["bold"]                  # (T, N) float32
    T, N = bold.shape
    if T < window_len + step:
        log.warning(
            "Subject %s: %d TRs is too short for window_len=%d + step=%d — skipping.",
            subject["subject_id"], T, window_len, step,
        )
        return None

    bold = zscore(bold)
    mean_fc = compute_fc(bold)
    bold_windows, fc_windows = sliding_fc_windows(bold, window_len=window_len, step=step)

    np.savez_compressed(
        out_path,
        bold=bold,
        mean_fc=mean_fc,
        bold_windows=bold_windows,
        fc_windows=fc_windows,
        label=np.int64(subject["label"]),
        subject_id=subject["subject_id"],
        site=subject["site"],
    )
    return out_path


def preprocess_all(
    subjects: list[dict],
    processed_dir: str | Path,
    window_len: int = 50,
    step: int = 5,
    overwrite: bool = False,
) -> list[Path]:
    """
    Preprocess all subjects, skipping those already cached.
    Returns list of successfully written .npz paths.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, subject in enumerate(subjects):
        path = preprocess_subject(
            subject, processed_dir,
            window_len=window_len, step=step, overwrite=overwrite,
        )
        if path is not None:
            paths.append(path)
        if (i + 1) % 50 == 0:
            log.info("Preprocessed %d / %d subjects.", i + 1, len(subjects))

    log.info("Preprocessing done: %d / %d subjects saved.", len(paths), len(subjects))
    return paths
