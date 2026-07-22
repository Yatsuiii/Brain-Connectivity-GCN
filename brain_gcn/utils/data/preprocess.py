"""
Preprocess ABIDE subjects into cached .npz files.

Each .npz contains:
    bold        (T, N)      — z-scored BOLD time series
    mean_fc     (N, N)      — full-scan Pearson FC
    bold_windows (W, N)      — std of BOLD per window (local signal power; node features)
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


def load_motion_params(subject: dict) -> np.ndarray | None:
    """Return the first six rigid-body motion columns, when available.

    The expected column order is three translations followed by three rotations
    in radians. Subjects without a two-dimensional ``confounds`` array containing
    at least six columns return ``None``.
    """
    confounds = subject.get("confounds")
    if confounds is None:
        return None

    params = np.asarray(confounds)
    if params.ndim != 2 or params.shape[1] < 6:
        return None
    return params[:, :6].astype(np.float32, copy=False)


def compute_fd(motion_params: np.ndarray, head_radius_mm: float = 50.0) -> np.ndarray:
    """Compute Power-style framewise displacement from rigid-body motion.

    Translational differences are measured in millimetres. Rotational
    differences, expected in radians, are converted to arc length using a
    spherical head radius of 50 mm by default. The result has length ``T - 1``.
    """
    params = np.asarray(motion_params, dtype=np.float32)
    if params.ndim != 2 or params.shape[1] < 6:
        raise ValueError("motion_params must have shape (T, 6) or more columns")
    if head_radius_mm <= 0:
        raise ValueError("head_radius_mm must be positive")

    delta = np.abs(np.diff(params[:, :6], axis=0))
    delta[:, 3:6] *= head_radius_mm
    return delta.sum(axis=1, dtype=np.float32)


def scrub_bold(
    bold: np.ndarray,
    fd: np.ndarray,
    fd_threshold: float = 0.5,
    min_clean_trs: int = 50,
) -> np.ndarray | None:
    """Remove frames whose preceding motion transition exceeds ``fd_threshold``.

    Frame zero is retained because it has no preceding transition. Returns
    ``None`` when fewer than ``min_clean_trs`` frames remain.
    """
    bold_array = np.asarray(bold)
    fd_array = np.asarray(fd).reshape(-1)
    if bold_array.ndim != 2:
        raise ValueError("bold must have shape (T, N)")
    if fd_array.shape[0] != max(0, bold_array.shape[0] - 1):
        raise ValueError("fd must have length T - 1")
    if fd_threshold < 0:
        raise ValueError("fd_threshold must be non-negative")
    if min_clean_trs < 1:
        raise ValueError("min_clean_trs must be at least 1")

    keep = np.ones(bold_array.shape[0], dtype=bool)
    keep[1:] = np.isfinite(fd_array) & (fd_array <= fd_threshold)
    cleaned = bold_array[keep]
    return cleaned if cleaned.shape[0] >= min_clean_trs else None


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
        window_bold=bold_windows,
        window_fc=fc_windows,
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
