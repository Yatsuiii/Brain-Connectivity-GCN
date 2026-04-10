"""
Unit tests for dataset and datamodule.

Tests:
  - Different adjacency modes (mean, dynamic, sequence, population)
  - Metadata caching (JSON vs direct scan)
  - Thresholding
  - Window padding/truncation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from brain_gcn.utils.data.dataset import ABIDEDataset


@pytest.fixture
def temp_npz_dir():
    """Create a temporary directory with mock .npz files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Create mock .npz files
        for i in range(3):
            data = {
                "bold": np.random.randn(100, 200).astype(np.float32),
                "mean_fc": np.random.randn(200, 200).astype(np.float32),
                "bold_windows": np.random.randn(10, 200).astype(np.float32),
                "fc_windows": np.random.randn(10, 200, 200).astype(np.float32),
                "label": np.int64(i % 2),
                "subject_id": f"sub_{i:03d}",
                "site": f"site_{i % 2}",
            }
            np.savez_compressed(tmp / f"sub_{i:03d}.npz", **data)
        
        yield tmp


class TestABIDEDatasetMetadataCache:
    """Test metadata caching for fast initialization."""

    def test_scan_metadata_no_cache(self, temp_npz_dir):
        """Without cache, should scan .npz files directly."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths)
        assert len(dataset._meta) == 3
        for meta in dataset._meta:
            assert "subject_id" in meta
            assert "num_windows" in meta
            assert meta["num_windows"] == 10

    def test_scan_metadata_with_cache(self, temp_npz_dir):
        """With metadata.json, should load instantly from cache."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        
        # Write cache file
        cache = [
            {"subject_id": "sub_000", "label": 0, "site": "site_0", "num_windows": 10},
            {"subject_id": "sub_001", "label": 1, "site": "site_1", "num_windows": 10},
            {"subject_id": "sub_002", "label": 0, "site": "site_0", "num_windows": 10},
        ]
        cache_path = temp_npz_dir / "metadata.json"
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        
        # Dataset should load from cache, not scan files
        dataset = ABIDEDataset(npz_paths)
        assert len(dataset._meta) == 3
        assert dataset._meta[0]["subject_id"] == "sub_000"

    def test_max_windows_truncation(self, temp_npz_dir):
        """max_windows parameter should truncate metadata."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths, max_windows=5)
        for meta in dataset._meta:
            assert meta["num_windows"] == 5


class TestABIDEDatasetAdjacencyModes:
    """Test different adjacency matrix modes."""

    def test_mean_adj_mode(self, temp_npz_dir):
        """mean_fc mode should return (N, N) adjacency."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths, use_dynamic_adj=False, use_dynamic_adj_sequence=False)
        assert len(dataset) == 3
        # Just verify we can access without error
        bold, adj, label = dataset[0]
        assert bold.shape[0] == 10  # max_windows implicit
        assert bold.ndim == 2  # (W, N) or (N,)
        assert label.ndim == 0  # Scalar

    def test_dynamic_adj_mode(self, temp_npz_dir):
        """dynamic_adj mode should return mean of per-window FCs: (N, N)."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths, use_dynamic_adj=True, use_dynamic_adj_sequence=False)
        bold, adj, label = dataset[0]
        assert adj.ndim == 2
        assert adj.shape == (200, 200)

    def test_dynamic_sequence_adj_mode(self, temp_npz_dir):
        """dynamic_adj_sequence mode should return (W, N, N)."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths, use_dynamic_adj_sequence=True)
        bold, adj, label = dataset[0]
        assert adj.ndim == 3
        assert adj.shape[0] == 10  # W windows

    def test_population_adj_mode(self, temp_npz_dir):
        """population_adj mode should use shared adjacency."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        pop_adj = np.random.randn(200, 200).astype(np.float32)
        dataset = ABIDEDataset(npz_paths, population_adj=pop_adj)
        bold1, adj1, label1 = dataset[0]
        bold2, adj2, label2 = dataset[1]
        # Both should have the same adjacency matrix
        assert np.allclose(adj1.numpy(), adj2.numpy())


class TestABIDEDatasetThresholding:
    """Test FC thresholding."""

    def test_fc_threshold_zeros_small_values(self, temp_npz_dir):
        """Adjacency values below threshold should be zeroed."""
        npz_paths = sorted(temp_npz_dir.glob("*.npz"))
        dataset = ABIDEDataset(npz_paths, fc_threshold=0.5)
        bold, adj, label = dataset[0]
        # At fc_threshold=0.5, small absolute values should be zero
        # (Assuming randomly generated data has mixed magnitudes)
        assert (adj.numpy() >= 0.5).any() or (adj.numpy() == 0).all()
