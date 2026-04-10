"""
PyTorch Dataset for preprocessed ABIDE subjects.

Each sample returns:
    bold_windows : (W, N)     — mean BOLD per ROI at each brain-state snapshot
    adj         : (N, N)     — adjacency for this subject
                               use_dynamic_adj=False → subject's mean FC
                               use_dynamic_adj=True  → mean of per-window FCs
                               use_population_adj=True → shared population adj
    label       : ()         — int64 scalar  (0 = TC, 1 = ASD)

The adjacency is left as raw (thresholded) FC values so the model can apply
its own Laplacian normalisation via utils.graph_conv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ABIDEDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[Path | str],
        population_adj: np.ndarray | None = None,
        use_dynamic_adj: bool = False,
        fc_threshold: float = 0.2,
        max_windows: int | None = None,
    ):
        """
        Parameters
        ----------
        npz_paths       : paths to per-subject .npz files from preprocess.py
        population_adj  : (N, N) pre-computed population-level adjacency.
                          If provided, every sample uses this shared adjacency.
        use_dynamic_adj : if True and population_adj is None, use mean of
                          per-window FCs; otherwise use mean_fc (full-scan FC).
        fc_threshold    : zero-out edges with |fc| < threshold before returning
        max_windows     : truncate all subjects to this many windows so that
                          batches have uniform seq_len (takes the first W windows)
        """
        self.npz_paths = [Path(p) for p in npz_paths]
        self.population_adj = (
            torch.FloatTensor(population_adj) if population_adj is not None else None
        )
        self.use_dynamic_adj = use_dynamic_adj
        self.fc_threshold = fc_threshold
        self.max_windows = max_windows

        # Pre-load labels + window counts for fast access without loading full arrays
        self._meta = self._scan_metadata()

    def _scan_metadata(self) -> list[dict]:
        meta = []
        for p in self.npz_paths:
            data = np.load(p, allow_pickle=True)
            W = data["bold_windows"].shape[0]
            if self.max_windows is not None:
                W = min(W, self.max_windows)
            meta.append(
                {
                    "label": int(data["label"]),
                    "subject_id": str(data["subject_id"]),
                    "site": str(data["site"]),
                    "num_windows": W,
                }
            )
        return meta

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int):
        data = np.load(self.npz_paths[idx], allow_pickle=True)

        # Node feature sequence: (W, N)
        bold_windows = data["bold_windows"].astype(np.float32)
        if self.max_windows is not None:
            bold_windows = bold_windows[: self.max_windows]

        # Adjacency: (N, N)
        if self.population_adj is not None:
            adj = self.population_adj
        else:
            if self.use_dynamic_adj:
                # Mean of per-window FCs
                wfc = data["fc_windows"].astype(np.float32)
                if self.max_windows is not None:
                    wfc = wfc[: self.max_windows]
                adj_np = wfc.mean(axis=0)
            else:
                adj_np = data["mean_fc"].astype(np.float32)

            # Threshold + absolute value (non-negative edge weights)
            adj_np = np.where(np.abs(adj_np) >= self.fc_threshold, np.abs(adj_np), 0.0)
            adj = torch.FloatTensor(adj_np)

        label = torch.tensor(int(data["label"]), dtype=torch.long)
        return torch.FloatTensor(bold_windows), adj, label

    # ------------------------------------------------------------------
    @property
    def labels(self) -> list[int]:
        return [m["label"] for m in self._meta]

    @property
    def num_nodes(self) -> int:
        data = np.load(self.npz_paths[0], allow_pickle=True)
        return data["mean_fc"].shape[0]

    @property
    def num_windows(self) -> int:
        return self._meta[0]["num_windows"]
