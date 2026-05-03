"""
PyTorch Lightning DataModule for ABIDE I.

Full pipeline (called once via prepare_data / setup):
  1. Download ABIDE via nilearn        (download.py)
  2. Preprocess subjects → .npz cache  (preprocess.py)
  3. Stratified train / val / test split
  4. Build population adjacency from training subjects  (functional_connectivity.py)
  5. Expose train / val / test DataLoaders

Usage:
    dm = ABIDEDataModule(data_dir="data", n_subjects=100)
    dm.prepare_data()
    dm.setup()
    for bold_windows, adj, label in dm.train_dataloader():
        ...
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from .dataset import ABIDEDataset
from .download import fetch_abide, extract_subjects
from .functional_connectivity import compute_population_adj
from .preprocess import preprocess_all

log = logging.getLogger(__name__)


def collate_fn(batch):
    """
    Custom collate: stack bold_windows, labels, and site_ids; keep adj as-is.
    Returns:
        bold_windows : (B, W, N)
        adj         : (B, N, N)
        labels      : (B,)
        site_ids    : (B,)
    """
    bold_windowss, adjs, labels, site_ids = zip(*batch)
    return (
        torch.stack(bold_windowss),
        torch.stack(adjs),
        torch.stack(labels),
        torch.stack(site_ids),
    )


class ABIDEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        n_subjects: int | None = None,
        window_len: int = 50,
        step: int = 5,
        max_windows: int | None = 30,
        fc_threshold: float = 0.2,
        use_dynamic_adj: bool = False,
        use_dynamic_adj_sequence: bool = False,
        use_population_adj: bool = True,
        preserve_fc_sign: bool = False,
        use_fc_variance: bool = False,
        use_fisher_z: bool = False,
        use_fc_degree_features: bool = False,
        use_fc_row_features: bool = False,
        n_pca_components: int = 0,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_strategy: str = "stratified",
        val_site: str | None = None,
        test_site: str | None = None,
        num_workers: int = 4,
        overwrite_cache: bool = False,
        force_prepare: bool = False,
    ):
        """
        Parameters
        ----------
        data_dir         : root directory for raw + processed data
        n_subjects       : cap for ABIDE download (None = all ~884)
        window_len       : sliding window length in TRs
        step             : sliding window step in TRs
        max_windows      : truncate each subject to this many windows
                           (ensures uniform batch shapes without padding)
        fc_threshold     : sparsify FC: zero edges with |fc| < threshold
        use_dynamic_adj  : per-subject: use mean of window FCs (vs. full-scan FC)
        use_dynamic_adj_sequence: per-subject: return one adjacency per window.
                           Ignored when use_population_adj=True.
        use_population_adj: compute a single population-level adj from training
                           set and use it for all subjects (recommended)
        batch_size       : samples per batch
        val_ratio        : fraction of data for validation
        test_ratio       : fraction of data for test
        split_strategy   : stratified random split or site_holdout split
        val_site         : validation site for site_holdout. If unset, chosen by size.
        test_site        : test site for site_holdout. If unset, largest site is used.
        num_workers      : DataLoader worker processes
        overwrite_cache  : re-preprocess even if .npz files exist
        force_prepare    : download/preprocess even when processed .npz files exist
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.n_subjects = n_subjects
        self.window_len = window_len
        self.step = step
        self.max_windows = max_windows
        self.fc_threshold = fc_threshold
        self.use_dynamic_adj = use_dynamic_adj
        self.use_dynamic_adj_sequence = use_dynamic_adj_sequence
        self.use_population_adj = use_population_adj
        self.preserve_fc_sign = preserve_fc_sign
        self.use_fc_variance = use_fc_variance
        self.use_fisher_z = use_fisher_z
        self.use_fc_degree_features = use_fc_degree_features
        self.use_fc_row_features = use_fc_row_features
        self.n_pca_components = n_pca_components
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_strategy = split_strategy
        self.val_site = val_site
        self.test_site = test_site
        self.num_workers = num_workers
        self.overwrite_cache = overwrite_cache
        self.force_prepare = force_prepare

        self._population_adj: np.ndarray | None = None
        self._site_fc_mean: dict[str, np.ndarray] = {}
        self._site_to_int: dict[str, int] = {}
        self._pca_mean: np.ndarray | None = None        # (D,) mean FC vector
        self._pca_components: np.ndarray | None = None  # (K, D) principal axes
        self._train_paths: list[Path] = []
        self._val_paths: list[Path] = []
        self._test_paths: list[Path] = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Download + preprocess (runs on rank 0 only in distributed settings)."""
        cached_paths = list(self.processed_dir.glob("*.npz"))
        n_cached = len(cached_paths)

        # Skip only when we already have enough subjects and no explicit override
        have_enough = (
            self.n_subjects is None or n_cached >= self.n_subjects
        )
        if cached_paths and have_enough and not self.overwrite_cache and not self.force_prepare:
            log.info(
                "Found %d cached subject files in %s; skipping download/preprocess.",
                n_cached,
                self.processed_dir,
            )
            return

        if n_cached > 0 and not self.overwrite_cache:
            log.info(
                "Have %d subjects, want %s — downloading remaining subjects.",
                n_cached,
                self.n_subjects or "all",
            )

        dataset = fetch_abide(
            data_dir=self.raw_dir,
            n_subjects=self.n_subjects,
        )
        subjects = extract_subjects(dataset, min_timepoints=self.window_len + self.step)
        preprocess_all(
            subjects,
            processed_dir=self.processed_dir,
            window_len=self.window_len,
            step=self.step,
            overwrite=self.overwrite_cache,
        )

    def setup(self, stage: str | None = None):
        """Build train/val/test splits and optionally the population adjacency."""
        all_paths = sorted(self.processed_dir.glob("*.npz"))
        if not all_paths:
            raise RuntimeError(
                f"No .npz files found in {self.processed_dir}. "
                "Run prepare_data() first."
            )

        # Read labels/sites for splitting
        labels = np.array([
            int(np.load(p, allow_pickle=True)["label"]) for p in all_paths
        ])
        sites = np.array([
            str(np.load(p, allow_pickle=True)["site"]) for p in all_paths
        ])

        # Build site → int mapping from ALL subjects (consistent across splits)
        self._site_to_int = {
            site: i for i, site in enumerate(sorted(set(sites.tolist())))
        }
        log.info("Sites (%d): %s", len(self._site_to_int), sorted(self._site_to_int))

        if self.split_strategy == "stratified":
            train_paths, val_paths, test_paths = self._stratified_split(
                all_paths, labels, self.val_ratio, self.test_ratio
            )
        elif self.split_strategy == "site_holdout":
            train_paths, val_paths, test_paths = self._site_holdout_split(
                all_paths, labels, sites, self.val_site, self.test_site
            )
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")
        self._train_paths = train_paths
        self._val_paths = val_paths
        self._test_paths = test_paths

        log.info(
            "Split (%s): train=%d  val=%d  test=%d",
            self.split_strategy,
            len(train_paths), len(val_paths), len(test_paths),
        )

        # Build population adjacency from training subjects only
        if self.use_population_adj:
            self._population_adj = self._build_population_adj(train_paths)

        # Compute per-site mean FC from training set (FC-domain site normalization)
        self._site_fc_mean = self._build_site_fc_mean(train_paths)

        # PCA on training FC upper triangles (reduces p>>n overfitting)
        if self.n_pca_components > 0:
            self._pca_mean, self._pca_components = self._build_pca(train_paths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset(self._train_paths),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset(self._val_paths),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset(self._test_paths),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Properties exposed to the model
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of ROIs (200 for cc200 atlas)."""
        data = np.load(self._train_paths[0], allow_pickle=True)
        return data["mean_fc"].shape[0]

    @property
    def num_windows(self) -> int:
        """Number of brain-state snapshots (sliding windows) per subject."""
        if self.max_windows is not None:
            return self.max_windows
        data = np.load(self._train_paths[0], allow_pickle=True)
        return data["bold_windows"].shape[0]

    @property
    def population_adj(self) -> np.ndarray | None:
        return self._population_adj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dataset(self, paths: list[Path]) -> ABIDEDataset:
        return ABIDEDataset(
            npz_paths=paths,
            population_adj=self._population_adj,
            use_dynamic_adj=self.use_dynamic_adj,
            use_dynamic_adj_sequence=self.use_dynamic_adj_sequence,
            fc_threshold=self.fc_threshold,
            max_windows=self.max_windows,
            site_fc_mean=self._site_fc_mean,
            preserve_fc_sign=self.preserve_fc_sign,
            site_to_int=self._site_to_int,
            use_fc_variance=self.use_fc_variance,
            use_fisher_z=self.use_fisher_z,
            pca_mean=self._pca_mean,
            pca_components=self._pca_components,
            use_fc_degree_features=self.use_fc_degree_features,
            use_fc_row_features=self.use_fc_row_features,
        )

    @property
    def num_sites(self) -> int:
        return len(self._site_to_int)

    @staticmethod
    def _stratified_split(
        paths: list[Path],
        labels: np.ndarray,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        paths = np.array(paths)

        # First split off test set
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
        train_val_idx, test_idx = next(sss_test.split(paths, labels))

        # Then split val from train
        val_size = val_ratio / (1.0 - test_ratio)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(sss_val.split(paths[train_val_idx], labels[train_val_idx]))

        return (
            list(paths[train_val_idx[train_idx]]),
            list(paths[train_val_idx[val_idx]]),
            list(paths[test_idx]),
        )

    @staticmethod
    def _site_holdout_split(
        paths: list[Path],
        labels: np.ndarray,
        sites: np.ndarray,
        val_site: str | None,
        test_site: str | None,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        paths_arr = np.array(paths)
        site_counts = Counter(sites.tolist())
        if len(site_counts) < 3:
            raise ValueError("site_holdout split needs at least 3 sites.")

        sorted_sites = [site for site, _ in site_counts.most_common()]
        if test_site is None:
            test_site = sorted_sites[1]
        if val_site is None:
            val_site = next((site for site in reversed(sorted_sites) if site != test_site), None)
        if val_site is None or val_site == test_site:
            raise ValueError("site_holdout split needs distinct val_site and test_site.")
        if test_site not in site_counts:
            raise ValueError(f"Unknown test_site '{test_site}'. Available: {sorted(site_counts)}")
        if val_site not in site_counts:
            raise ValueError(f"Unknown val_site '{val_site}'. Available: {sorted(site_counts)}")

        train_mask = (sites != val_site) & (sites != test_site)
        val_mask = sites == val_site
        test_mask = sites == test_site

        ABIDEDataModule._assert_both_labels(labels[train_mask], "train")
        ABIDEDataModule._assert_both_labels(labels[val_mask], "val")
        ABIDEDataModule._assert_both_labels(labels[test_mask], "test")

        return (
            list(paths_arr[train_mask]),
            list(paths_arr[val_mask]),
            list(paths_arr[test_mask]),
        )

    @staticmethod
    def _assert_both_labels(labels: np.ndarray, split_name: str) -> None:
        unique = set(labels.tolist())
        if unique != {0, 1}:
            raise ValueError(
                f"{split_name} split must contain both labels, got {sorted(unique)}."
            )

    def _build_pca(self, train_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        """Compute PCA on training-set FC upper triangles using truncated SVD.

        Returns
        -------
        mean_vec   : (D,)   mean FC vector (for centering)
        components : (K, D) top-K principal axes (rows = PCs)

        With D=19900 features and N≈660 training subjects, PCA reduces to the
        N-1 dimensional subspace anyway. Using K<<N avoids p>>n overfitting:
        the MLP trains on K features rather than 19900.
        """
        K = self.n_pca_components
        log.info("Computing PCA (K=%d) from %d training FC matrices ...", K, len(train_paths))

        # Build training matrix: (N_train, D)
        rows = []
        for p in train_paths:
            data = np.load(p, allow_pickle=True)
            fc = data["mean_fc"].astype(np.float32)
            n = fc.shape[0]
            r, c = np.triu_indices(n, k=1)
            if self.use_fisher_z:
                fc = np.arctanh(np.clip(fc, -0.9999, 0.9999))
            rows.append(fc[r, c])

        X = np.stack(rows, axis=0)          # (N_train, D)
        mean_vec = X.mean(axis=0)           # (D,)
        X_centered = X - mean_vec           # (N_train, D)

        # Truncated SVD via economy SVD on the smaller dimension
        # X = U S Vt  →  principal components = Vt[:K]
        # Since N << D, use X @ Xt for the eigen-decomposition shortcut
        # (N_train × N_train covariance, then recover Vt)
        C = (X_centered @ X_centered.T) / (len(train_paths) - 1)   # (N, N)
        eigenvalues, U = np.linalg.eigh(C)                          # ascending
        # eigh returns ascending; we want descending
        idx = np.argsort(-eigenvalues)
        U = U[:, idx[:K]]                                            # (N, K)
        components = (X_centered.T @ U)                              # (D, K)
        # Normalise each column to unit length → rows of Vt
        components /= np.linalg.norm(components, axis=0, keepdims=True) + 1e-8
        components = components.T.astype(np.float32)                 # (K, D)

        var_explained = eigenvalues[idx[:K]].sum() / (eigenvalues.sum() + 1e-8)
        log.info("PCA: top-%d components explain %.1f%% of FC variance.", K, 100 * var_explained)
        return mean_vec.astype(np.float32), components

    def _build_site_fc_mean(self, train_paths: list[Path]) -> dict[str, np.ndarray]:
        """Compute per-site mean FC matrix (N, N) from training subjects.
        Subtracting this at load time removes scanner-specific connectivity biases
        (a simple FC-domain site normalization). BOLD is already z-scored so
        BOLD-domain corrections have no effect."""
        log.info("Computing per-site FC means from %d training subjects ...", len(train_paths))
        site_sums: dict[str, np.ndarray] = {}
        site_counts: dict[str, int] = {}
        for p in train_paths:
            data = np.load(p, allow_pickle=True)
            site = str(data["site"])
            fc = data["mean_fc"].astype(np.float32)   # (N, N)
            if site not in site_sums:
                site_sums[site] = np.zeros_like(fc)
                site_counts[site] = 0
            site_sums[site] += fc
            site_counts[site] += 1
        return {s: site_sums[s] / site_counts[s] for s in site_sums}

    def _build_population_adj(self, train_paths: list[Path]) -> np.ndarray:
        log.info("Building population adjacency from %d training subjects ...", len(train_paths))
        mean_fcs = []
        for p in train_paths:
            data = np.load(p, allow_pickle=True)
            mean_fcs.append(data["mean_fc"].astype(np.float32))
        adj = compute_population_adj(mean_fcs, threshold=self.fc_threshold)
        log.info(
            "Population adj: %d nodes, %.1f%% edges non-zero.",
            adj.shape[0],
            100.0 * (adj > 0).sum() / adj.size,
        )
        return adj

    # ------------------------------------------------------------------
    # argparse integration
    # ------------------------------------------------------------------

    @staticmethod
    def add_data_specific_arguments(parent_parser: argparse.ArgumentParser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="data")
        parser.add_argument("--n_subjects", type=int, default=None)
        parser.add_argument("--window_len", type=int, default=50)
        parser.add_argument("--step", type=int, default=5)
        parser.add_argument("--max_windows", type=int, default=30)
        parser.add_argument("--fc_threshold", type=float, default=0.2)
        parser.add_argument("--use_dynamic_adj", action="store_true")
        parser.add_argument("--use_dynamic_adj_sequence", action="store_true")
        parser.add_argument("--use_population_adj", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--preserve_fc_sign", action="store_true",
                            help="Keep signed FC values in adjacency (required for fc_mlp).")
        parser.add_argument("--use_fc_variance", action="store_true",
                            help="Append std(fc_windows) as a second feature channel alongside mean FC.")
        parser.add_argument("--use_fc_degree_features", action="store_true",
                            help="Replace BOLD std node features with per-ROI mean |FC| per window.")
        parser.add_argument("--use_fc_row_features", action="store_true",
                            help="Use FC rows as node features (W,N,N). Requires graph_temporal + in_features=num_nodes.")
        parser.add_argument("--use_fisher_z", action="store_true",
                            help="Apply Fisher r-to-z transform to FC values before classification.")
        parser.add_argument("--n_pca_components", type=int, default=0,
                            help="If >0, reduce FC to this many PCA components before the MLP.")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--val_ratio", type=float, default=0.1)
        parser.add_argument("--test_ratio", type=float, default=0.1)
        parser.add_argument("--split_strategy", choices=["stratified", "site_holdout"], default="stratified")
        parser.add_argument("--val_site", type=str, default=None)
        parser.add_argument("--test_site", type=str, default=None)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument(
            "--overwrite_cache",
            action="store_true",
            help="Force re-download and re-preprocess even if .npz files already exist.",
        )
        return parser
