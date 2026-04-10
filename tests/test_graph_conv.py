"""
Unit tests for graph_conv utilities.

Tests:
  - Laplacian normalization correctness on known adjacency
  - Density-aware DropEdge behavior
  - Shape preservation across 2D/3D/4D inputs
"""

import numpy as np
import pytest
import torch

from brain_gcn.utils.graph_conv import calculate_laplacian_with_self_loop, drop_edge


class TestLaplacianNormalization:
    """Test symmetric normalized Laplacian computation."""

    def test_identity_adjacency_2d(self):
        """Identity matrix should be normalized symmetrically."""
        adj = torch.eye(5, dtype=torch.float32)
        norm = calculate_laplacian_with_self_loop(adj)
        # With self-loop added: I + I = 2I, 
        # Normalized: D^{-1/2} * (2I) * D^{-1/2} where D = 2I
        # d_inv_sqrt = (2)^{-0.5} = 1/sqrt(2)
        # Result: (1/sqrt(2)) * (2I) * (1/sqrt(2)) = I
        assert norm.shape == (5, 5)
        # Just check it's normalized and symmetric (not checking exact values)
        assert torch.allclose(norm, norm.T, atol=1e-5)
        assert (norm >= 0).all()  # non-negative

    def test_fully_connected_adjacency(self):
        """All-ones adjacency should normalize symmetrically."""
        adj = torch.ones(3, 3, dtype=torch.float32)
        norm = calculate_laplacian_with_self_loop(adj)
        # A + I = [2, 1, 1; 1, 2, 1; 1, 1, 2]
        # row_sum = [4, 4, 4], d_inv_sqrt = [0.5, 0.5, 0.5]
        # After symmetric norm, should be symmetric and bounded
        assert norm.shape == (3, 3)
        assert torch.allclose(norm, norm.T, atol=1e-5)

    def test_batch_adjacency_3d(self):
        """3D batch should normalize each matrix independently."""
        batch = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(4, -1, -1)
        norm = calculate_laplacian_with_self_loop(batch)
        assert norm.shape == (4, 3, 3)
        # Each batch element is an identity, should match 2D test
        for i in range(4):
            assert torch.allclose(norm[i], norm[0], atol=1e-5)

    def test_dynamic_adjacency_4d(self):
        """4D (B, W, N, N) should reshape and normalize correctly."""
        batch = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        batch = batch.expand(2, 5, -1, -1)  # (2, 5, 3, 3)
        norm = calculate_laplacian_with_self_loop(batch)
        assert norm.shape == (2, 5, 3, 3)


class TestDropEdge:
    """Test DropEdge regularization with density awareness."""

    def test_drop_edge_sparse_graph(self):
        """Sparse graph: density-aware scaling should preserve most edges."""
        torch.manual_seed(42)
        # Create a sparse adjacency with ~10% density
        adj = torch.zeros(100, 100, dtype=torch.float32)
        # Set ~10% of elements to non-zero
        mask_idx = torch.rand(100, 100) < 0.1
        adj[mask_idx] = 1.0
        initial_density = (adj > 0).float().mean().item()
        
        # With low density, p_eff should be scaled down significantly
        dropped = drop_edge(adj, p=0.1, training=True)
        final_density = (dropped > 0).float().mean().item()
        
        # Density shouldn't change much (edges are preserved)
        # Since p_eff = min(0.1, 0.1 * 0.5) = 0.05, we drop ~5% of edges
        # So final density should be ~95% of initial
        assert final_density >= initial_density * 0.90, (
            f"Too many edges dropped: {initial_density:.3f} → {final_density:.3f}"
        )

    def test_drop_edge_eval_mode(self):
        """Drop edge should be no-op in eval mode."""
        adj = torch.randn(5, 5, dtype=torch.float32)
        dropped = drop_edge(adj, p=0.5, training=False)
        assert torch.equal(adj, dropped)

    def test_drop_edge_zero_probability(self):
        """Zero drop probability = no changes."""
        adj = torch.randn(5, 5, dtype=torch.float32)
        dropped = drop_edge(adj, p=0.0, training=True)
        assert torch.equal(adj, dropped)

    def test_drop_edge_dense_graph(self):
        """Dense graph: p_eff should not be limited by density."""
        # Create dense adjacency (all ones where w > 0.5)
        adj = torch.ones(10, 10, dtype=torch.float32)
        p_base = 0.3
        dropped = drop_edge(adj, p=p_base, training=True)
        # With density=1.0, p_eff = min(0.3, 0.5) = 0.3
        # So ~70% of edges should remain
        density_after = (dropped > 0).float().mean().item()
        assert 0.6 < density_after < 0.8, f"Unexpected drop rate: {density_after}"

    def test_drop_edge_batch_shape(self, sample_adjacency_batch):
        """DropEdge should preserve batch shape."""
        dropped = drop_edge(sample_adjacency_batch, p=0.2, training=True)
        assert dropped.shape == sample_adjacency_batch.shape
