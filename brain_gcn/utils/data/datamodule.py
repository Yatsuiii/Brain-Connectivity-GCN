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
    Custom collate: stack bold_windows and labels; keep adj as-is (all same shape).
    Returns:
        bold_windows : (B, W, N)
        adj         : (B, N, N)
        labels      : (B,)
    """
    bold_windowss, adjs, labels = zip(*batch)
    return (
        torch.stack(bold_windowss),
        torch.stack(adjs),
        torch.stack(labels),
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
        self._train_paths: list[Path] = []
        self._val_paths: list[Path] = []
        self._test_paths: list[Path] = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Download + preprocess (runs on rank 0 only in distributed settings)."""
        cached_paths = list(self.processed_dir.glob("*.npz"))
        if cached_paths and not self.overwrite_cache and not self.force_prepare:
            log.info(
                "Found %d cached subject files in %s; skipping download/preprocess.",
                len(cached_paths),
                self.processed_dir,
            )
            return

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
        )

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
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--val_ratio", type=float, default=0.1)
        parser.add_argument("--test_ratio", type=float, default=0.1)
        parser.add_argument("--split_strategy", choices=["stratified", "site_holdout"], default="stratified")
        parser.add_argument("--val_site", type=str, default=None)
        parser.add_argument("--test_site", type=str, default=None)
        parser.add_argument("--num_workers", type=int, default=4)
        return parser
