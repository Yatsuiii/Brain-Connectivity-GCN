"""
Functional connectivity computation and sliding-window decomposition.

For each subject we produce:
  - mean_fc      (num_rois, num_rois)  — Pearson correlation over full scan
  - bold_windows  (num_windows, num_rois) — mean BOLD per ROI per window
  - fc_windows    (num_windows, num_rois, num_rois) — per-window Pearson FC

bold_windows is the node-feature sequence fed into the BrainGCN encoder
(one scalar per ROI per brain-state snapshot).  fc_windows is the dynamic
adjacency sequence (how connectivity evolves across windows).
mean_fc is an alternative static adjacency (averaged across the full scan).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Full-scan FC
# ---------------------------------------------------------------------------

def compute_fc(bold: np.ndarray) -> np.ndarray:
    """
    Pearson correlation matrix for a single subject.

    Parameters
    ----------
    bold : (T, N)

    Returns
    -------
    fc : (N, N) float32, values in [-1, 1]
    """
    # np.corrcoef expects (N, T)
    fc = np.corrcoef(bold.T).astype(np.float32)
    # Replace NaN (zero-variance ROIs) with 0
    np.nan_to_num(fc, copy=False)
    return fc


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def sliding_fc_windows(
    bold: np.ndarray,
    window_len: int = 50,
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a BOLD time series into overlapping windows and compute per-window
    Pearson FC and mean BOLD.

    Parameters
    ----------
    bold       : (T, N) float32
    window_len : number of TRs per window (default 50 ≈ 100 s at TR=2s)
    step       : stride between windows in TRs (default 5)

    Returns
    -------
    bold_windows : (W, N)       mean BOLD per ROI per window
    fc_windows   : (W, N, N)   Pearson FC per window

    where W = number of windows = (T - window_len) // step + 1
    """
    T, N = bold.shape
    starts = range(0, T - window_len + 1, step)
    W = len(starts)

    bold_windows = np.empty((W, N), dtype=np.float32)
    fc_windows = np.empty((W, N, N), dtype=np.float32)

    for i, s in enumerate(starts):
        segment = bold[s : s + window_len]          # (window_len, N)
        bold_windows[i] = segment.mean(axis=0)       # (N,)
        fc_windows[i] = compute_fc(segment)          # (N, N)

    return bold_windows, fc_windows


# ---------------------------------------------------------------------------
# FC post-processing
# ---------------------------------------------------------------------------

def threshold_fc(
    fc: np.ndarray,
    threshold: float | None = None,
    keep_top_k: int | None = None,
    absolute: bool = True,
) -> np.ndarray:
    """
    Sparsify an FC matrix to reduce noise.

    One of `threshold` or `keep_top_k` must be provided.

    Parameters
    ----------
    fc          : (..., N, N)
    threshold   : zero-out values with |fc| < threshold
    keep_top_k  : keep top-k connections per node (symmetric, per-row)
    absolute    : use |fc| for comparison (keeps negative correlations)

    Returns
    -------
    Thresholded FC with the same shape as input.
    """
    fc = fc.copy()
    if threshold is not None:
        mask = (np.abs(fc) if absolute else fc) < threshold
        fc[mask] = 0.0
    elif keep_top_k is not None:
        # Apply per-row top-k independently
        original_shape = fc.shape
        fc_2d = fc.reshape(-1, original_shape[-1])  # (...*N, N)
        vals = np.abs(fc_2d) if absolute else fc_2d
        kth = np.partition(vals, -keep_top_k, axis=-1)[:, -keep_top_k : -keep_top_k + 1]
        mask = vals < kth
        fc_2d[mask] = 0.0
        fc = fc_2d.reshape(original_shape)
    else:
        raise ValueError("Provide either `threshold` or `keep_top_k`.")
    return fc


def normalize_fc(fc: np.ndarray) -> np.ndarray:
    """
    Min-max normalize FC values to [0, 1] for use as edge weights.
    Operates on the last two dimensions (N, N).
    """
    fc = fc.copy()
    mn, mx = fc.min(), fc.max()
    if mx > mn:
        fc = (fc - mn) / (mx - mn)
    return fc.astype(np.float32)


# ---------------------------------------------------------------------------
# Population-level static adjacency
# ---------------------------------------------------------------------------

def compute_population_adj(
    mean_fcs: list[np.ndarray],
    threshold: float = 0.2,
    absolute: bool = True,
) -> np.ndarray:
    """
    Build a single population-level adjacency by averaging per-subject mean FCs
    and thresholding.

    Parameters
    ----------
    mean_fcs  : list of (N, N) arrays — one per subject
    threshold : zero-out edges with |mean_fc| < threshold

    Returns
    -------
    adj : (N, N) float32 — binary or weighted adjacency
    """
    pop_fc = np.mean(np.stack(mean_fcs, axis=0), axis=0)  # (N, N)
    adj = threshold_fc(pop_fc, threshold=threshold, absolute=absolute)
    # Make non-negative (GCN typically expects non-negative adjacency)
    adj = np.abs(adj).astype(np.float32)
    return adj
