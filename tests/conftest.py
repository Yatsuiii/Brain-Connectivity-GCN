"""
Shared fixtures for Brain-Connectivity-GCN tests.
"""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_bold():
    """Generate sample BOLD data: (T, N) = (200, 50 ROIs)."""
    return np.random.randn(200, 50).astype(np.float32)


@pytest.fixture
def sample_bold_batch():
    """Generate batched BOLD data: (B, W, N) = (4, 20, 50)."""
    return torch.randn(4, 20, 50, dtype=torch.float32)


@pytest.fixture
def sample_adjacency():
    """Generate sample adjacency matrix: (N, N) = (50, 50)."""
    # Dense graph with some zeros
    adj = np.random.randn(50, 50).astype(np.float32)
    adj = (adj + adj.T) / 2  # symmetric
    adj = np.abs(adj)  # non-negative
    adj[adj < 0.5] = 0  # sparse
    return adj


@pytest.fixture
def sample_adjacency_batch():
    """Generate batched adjacency matrix: (B, N, N) = (4, 50, 50)."""
    return torch.from_numpy(
        np.array([np.random.randn(50, 50).astype(np.float32) for _ in range(4)])
    )
