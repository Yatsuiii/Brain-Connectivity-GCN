"""
PyTorch Dataset for preprocessed ABIDE subjects.

Each sample returns:
    bold_windows : (W, N)     — mean BOLD per ROI at each brain-state snapshot
    adj         : (N, N) or (W, N, N) — adjacency for this subject
                               use_dynamic_adj=False → subject's mean FC
                               use_dynamic_adj=True  → mean of per-window FCs
                               use_dynamic_adj_sequence=True → per-window FCs
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
        use_dynamic_adj_sequence: bool = False,
        fc_threshold: float = 0.2,
        max_windows: int | None = None,
        site_fc_mean: dict[str, np.ndarray] | None = None,
        preserve_fc_sign: bool = False,
        site_to_int: dict[str, int] | None = None,
        use_fc_variance: bool = False,
        use_fisher_z: bool = False,
        pca_mean: np.ndarray | None = None,
        pca_components: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        npz_paths       : paths to per-subject .npz files from preprocess.py
        population_adj  : (N, N) pre-computed population-level adjacency.
                          If provided, every sample uses this shared adjacency.
        use_dynamic_adj : if True and population_adj is None, use mean of
                          per-window FCs; otherwise use mean_fc (full-scan FC).
        use_dynamic_adj_sequence : if True and population_adj is None, return
                          per-window FCs with shape (W, N, N).
        fc_threshold    : zero-out edges with |fc| < threshold before returning
        max_windows     : truncate all subjects to this many windows so that
                          batches have uniform seq_len (takes the first W windows)
        site_fc_mean    : per-site mean FC matrix (N, N) computed from training
                          set. Subtracted from each subject's FC before thresholding
                          to remove scanner/site connectivity biases (FC-domain
                          site normalization). BOLD is already z-scored so
                          BOLD-domain corrections have no effect.
        preserve_fc_sign: if True, keep signed FC values in the adjacency instead
                          of converting to |FC|. Required for fc_mlp which uses
                          signed correlations as direct features (anti-correlations
                          between brain networks are diagnostically relevant).
        """
        self.npz_paths = [Path(p) for p in npz_paths]
        self.population_adj = (
            torch.FloatTensor(population_adj) if population_adj is not None else None
        )
        self.use_dynamic_adj = use_dynamic_adj
        self.use_dynamic_adj_sequence = use_dynamic_adj_sequence
        self.fc_threshold = fc_threshold
        self.max_windows = max_windows
        self.site_fc_mean = site_fc_mean or {}
        self.preserve_fc_sign = preserve_fc_sign
        self.site_to_int = site_to_int or {}
        self.use_fc_variance = use_fc_variance
        self.use_fisher_z = use_fisher_z
        self.pca_mean = pca_mean
        self.pca_components = pca_components

        # Pre-load labels + window counts for fast access without loading full arrays
        self._meta = self._scan_metadata()

    @staticmethod
    def _array(data: np.lib.npyio.NpzFile, primary: str, legacy: str) -> np.ndarray:
        if primary in data:
            return data[primary]
        if legacy in data:
            return data[legacy]
        raise KeyError(f"Expected '{primary}' or legacy '{legacy}' in subject archive")

    def _threshold(self, adj_np: np.ndarray, preserve_sign: bool = False) -> np.ndarray:
        mask = np.abs(adj_np) >= self.fc_threshold
        if preserve_sign:
            return np.where(mask, adj_np, 0.0)
        return np.where(mask, np.abs(adj_np), 0.0)

    @staticmethod
    def _fisher_z(fc: np.ndarray) -> np.ndarray:
        """Fisher's r-to-z transform: z = arctanh(r).

        Linearises the correlation space — correlations near ±1 are compressed
        in Pearson space but uniform in z-space. Stabilises variance across
        different correlation magnitudes, which matters for linear classifiers.
        Clipped to ±0.9999 to avoid ±inf at perfect correlations.
        """
        return np.arctanh(np.clip(fc, -0.9999, 0.9999))

    @staticmethod
    def _pad_or_truncate_windows(array: np.ndarray, max_windows: int | None) -> np.ndarray:
        if max_windows is None:
            return array
        if array.shape[0] >= max_windows:
            return array[:max_windows]
        pad_count = max_windows - array.shape[0]
        pad = np.repeat(array[-1:], pad_count, axis=0)
        return np.concatenate([array, pad], axis=0)

    def _scan_metadata(self) -> list[dict]:
        meta = []
        for p in self.npz_paths:
            data = np.load(p, allow_pickle=True)
            W = self._array(data, "bold_windows", "window_bold").shape[0]
            if self.max_windows is not None:
                W = self.max_windows
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
        bold_windows = self._array(data, "bold_windows", "window_bold").astype(np.float32)
        bold_windows = self._pad_or_truncate_windows(bold_windows, self.max_windows)

        site = str(data["site"])

        # Adjacency
        if self.population_adj is not None:
            adj = self.population_adj                          # (N, N) shared

        elif self.use_dynamic_adj_sequence:
            wfc = self._array(data, "fc_windows", "window_fc").astype(np.float32)
            wfc = self._pad_or_truncate_windows(wfc, self.max_windows)
            if site in self.site_fc_mean:
                wfc = wfc - self.site_fc_mean[site].astype(np.float32)[None]
            adj = torch.FloatTensor(
                self._threshold(wfc, self.preserve_fc_sign).astype(np.float32)
            )                                                  # (W, N, N)

        elif self.use_dynamic_adj:
            wfc = self._array(data, "fc_windows", "window_fc").astype(np.float32)
            wfc = self._pad_or_truncate_windows(wfc, self.max_windows)
            fc = wfc.mean(axis=0)
            if site in self.site_fc_mean:
                fc = fc - self.site_fc_mean[site].astype(np.float32)
            adj = torch.FloatTensor(
                self._threshold(fc, self.preserve_fc_sign).astype(np.float32)
            )                                                  # (N, N)

        else:
            # Static per-subject mean FC
            mean_np = data["mean_fc"].astype(np.float32)
            if site in self.site_fc_mean:
                mean_np = mean_np - self.site_fc_mean[site].astype(np.float32)
            if self.use_fisher_z:
                mean_np = self._fisher_z(mean_np)
            mean_np = self._threshold(mean_np, self.preserve_fc_sign).astype(np.float32)

            if self.pca_mean is not None and self.pca_components is not None:
                # PCA projection: (D,) → (K,)
                # Extract upper triangle the same way the MLP model does
                n = mean_np.shape[0]
                r, c = np.triu_indices(n, k=1)
                x_vec = mean_np[r, c] - self.pca_mean               # centre
                x_pca = (self.pca_components @ x_vec).astype(np.float32)  # (K,)
                # Return as (1, K) so collate_fn stacks to (B, 1, K); model flattens
                adj = torch.FloatTensor(x_pca).unsqueeze(0)          # (1, K)

            elif self.use_fc_variance:
                # Second channel: temporal std of FC — captures connection instability
                wfc = self._array(data, "fc_windows", "window_fc").astype(np.float32)
                wfc = self._pad_or_truncate_windows(wfc, self.max_windows)
                std_np = wfc.std(axis=0).astype(np.float32)
                adj = torch.FloatTensor(np.stack([mean_np, std_np], axis=0))  # (2, N, N)

            else:
                adj = torch.FloatTensor(mean_np)               # (N, N)

        label = torch.tensor(int(data["label"]), dtype=torch.long)
        site_id = torch.tensor(self.site_to_int.get(site, -1), dtype=torch.long)
        return torch.FloatTensor(bold_windows), adj, label, site_id

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
