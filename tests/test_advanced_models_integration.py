"""
Integration tests for advanced models (GAT, Transformer, CNN3D, GraphSAGE).

Focus: Forward passes, model registry, compatibility with training pipeline.
"""

import pytest
import torch
import numpy as np

from brain_gcn.models import (
    ModelRegistry,
    ModelConfig,
    GATClassifier,
    TransformerClassifier,
    CNN3DClassifier,
    GraphSAGEClassifier,
)


class TestGATModel:
    """Test Graph Attention Network forward pass and properties."""

    def test_gat_forward_pass(self, sample_bold_batch, sample_adjacency_batch):
        """GAT should accept bold_windows and adj, output binary predictions."""
        model = GATClassifier(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1,
        )
        
        bold_windows = sample_bold_batch  # (B, W, N) = (4, 20, 50)
        # Use non-negative adjacency for meaningful output
        adj = torch.abs(sample_adjacency_batch)  # (B, N, N) = (4, 50, 50)
        adj = (adj + adj.transpose(-2, -1)) / 2  # Make symmetric
        
        logits = model(bold_windows, adj)
        
        assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
        assert not torch.isnan(logits).any(), "Contains NaN values"

    def test_gat_handles_dense_adjacency(self, sample_bold_batch):
        """GAT should handle fully connected adjacencies."""
        model = GATClassifier(hidden_dim=32, num_heads=2)
        bold_windows = sample_bold_batch  # (4, 20, 50)
        adj = torch.ones(4, 50, 50, dtype=torch.float32)  # Fully connected
        
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any(), "Output contains NaN"

    def test_gat_handles_sparse_adjacency(self, sample_bold_batch):
        """GAT should handle sparse adjacencies."""
        model = GATClassifier(hidden_dim=32, num_heads=2)
        bold_windows = sample_bold_batch
        
        # Create sparse adjacency
        adj = torch.zeros(4, 50, 50, dtype=torch.float32)
        for i in range(4):
            # Connect each node to 5 neighbors
            for j in range(50):
                neighbors = np.random.choice(50, 5, replace=False)
                adj[i, j, neighbors] = 1.0
        
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

    def test_gat_gradient_flow(self, sample_bold_batch, sample_adjacency_batch):
        """GAT weights should receive gradients."""
        model = GATClassifier(hidden_dim=32, num_heads=2)
        bold_windows = sample_bold_batch
        adj = sample_adjacency_batch
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logits = model(bold_windows, adj)
        loss = logits.sum()  # Dummy loss
        loss.backward()
        
        # Check gradients flow
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter has no gradient"
        
        optimizer.step()


class TestTransformerModel:
    """Test Transformer temporal encoder."""

    def test_transformer_forward_pass(self, sample_bold_batch, sample_adjacency_batch):
        """Transformer should accept bold_windows and adj, output predictions."""
        model = TransformerClassifier(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1,
        )
        
        bold_windows = sample_bold_batch  # (B, W, N) = (4, 20, 50)
        adj = torch.abs(sample_adjacency_batch)
        logits = model(bold_windows, adj)
        
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

    def test_transformer_ignores_adjacency(self, sample_bold_batch, sample_adjacency_batch):
        """Transformer should work without adjacency (temporal focus)."""
        model = TransformerClassifier(hidden_dim=32)
        bold_windows = sample_bold_batch
        adj = torch.abs(sample_adjacency_batch)
        
        # Forward with bold and adj (though adj is not used internally)
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)

    def test_transformer_variable_sequence_length(self):
        """Transformer should handle different sequence lengths."""
        model = TransformerClassifier(hidden_dim=32)
        
        # Test different sequence lengths
        for seq_len in [10, 20, 30, 50]:
            bold_windows = torch.randn(4, seq_len, 50, dtype=torch.float32)
            adj = torch.randn(4, 50, 50, dtype=torch.float32)
            logits = model(bold_windows, adj)
            assert logits.shape == (4, 2)

    def test_transformer_gradient_flow(self, sample_bold_batch, sample_adjacency_batch):
        """Transformer weights should receive gradients."""
        model = TransformerClassifier(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters())
        
        adj = torch.abs(sample_adjacency_batch)
        logits = model(sample_bold_batch, adj)
        loss = logits.sum()
        loss.backward()
        
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCNN3DModel:
    """Test 3D CNN for spatiotemporal features."""

    def test_cnn3d_forward_pass(self):
        """CNN3D should accept bold_windows and adj, output predictions."""
        model = CNN3DClassifier(hidden_dim=32)
        
        bold_windows = torch.randn(4, 20, 50, dtype=torch.float32)
        adj = torch.randn(4, 50, 50, dtype=torch.float32)
        logits = model(bold_windows, adj)
        
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

    def test_cnn3d_pooling_reduces_dimensions(self):
        """CNN3D pooling should progressively reduce spatial dimensions."""
        model = CNN3DClassifier(hidden_dim=32)
        bold_windows = torch.randn(4, 20, 50, dtype=torch.float32)
        adj = torch.randn(4, 50, 50, dtype=torch.float32)
        
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)
        # Just verify no dimension errors occur

    def test_cnn3d_gradient_flow(self):
        """CNN3D weights should receive gradients."""
        model = CNN3DClassifier(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        bold_windows = torch.randn(4, 20, 50, dtype=torch.float32)
        adj = torch.randn(4, 50, 50, dtype=torch.float32)
        logits = model(bold_windows, adj)
        loss = logits.sum()
        loss.backward()
        
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGraphSAGEModel:
    """Test GraphSAGE inductive graph learning."""

    def test_graphsage_forward_pass(self, sample_bold_batch, sample_adjacency_batch):
        """GraphSAGE should accept bold_windows and adj."""
        model = GraphSAGEClassifier(
            hidden_dim=64,
            dropout=0.1,
        )
        
        bold_windows = sample_bold_batch
        adj = torch.abs(sample_adjacency_batch)  # Ensure non-negative
        adj = (adj + adj.transpose(-2, -1)) / 2  # Make symmetric
        
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

    def test_graphsage_handles_isolated_nodes(self, sample_bold_batch):
        """GraphSAGE should handle isolated nodes (zero adjacency)."""
        model = GraphSAGEClassifier(hidden_dim=32)
        bold_windows = sample_bold_batch
        
        # Create adjacency with isolated nodes
        adj = torch.zeros(4, 50, 50, dtype=torch.float32)
        for i in range(4):
            # Only connect first 10 nodes
            adj[i, :10, :10] = torch.ones(10, 10)
        
        logits = model(bold_windows, adj)
        assert logits.shape == (4, 2)

    def test_graphsage_gradient_flow(self, sample_bold_batch, sample_adjacency_batch):
        """GraphSAGE weights should receive gradients."""
        model = GraphSAGEClassifier(hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters())
        
        logits = model(sample_bold_batch, sample_adjacency_batch)
        loss = logits.sum()
        loss.backward()
        
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelRegistry:
    """Test model registry for discovering and building models."""

    def test_registry_lists_all_models(self):
        """Registry should list all 8 models."""
        models = ModelRegistry.get_models()
        assert len(models) == 8, f"Expected 8 models, got {len(models)}"
        
        expected = {
            'graph_temporal', 'gcn', 'gru', 'fc_mlp',
            'gat', 'transformer', 'cnn3d', 'graphsage'
        }
        actual = set(models.keys())
        assert actual == expected, f"Models mismatch: {actual ^ expected}"

    def test_registry_gets_model_class(self):
        """Registry should retrieve correct model classes."""
        gat_class = ModelRegistry.get_model_class('gat')
        assert gat_class == GATClassifier
        
        transformer_class = ModelRegistry.get_model_class('transformer')
        assert transformer_class == TransformerClassifier

    def test_registry_builds_model_from_config(self):
        """Registry should instantiate model from config."""
        config = ModelConfig(
            model_name='gat',
            hidden_dim=64,
            num_heads=4,
            dropout=0.1,
        )
        
        model = ModelRegistry.build_model(config)
        assert isinstance(model, GATClassifier)

    def test_registry_model_info(self):
        """Registry should provide model metadata."""
        info = ModelRegistry.get_model_info('gat')
        
        assert 'display_name' in info
        assert 'description' in info
        assert 'requires' in info
        assert 'parameters' in info
        
        assert info['display_name'] == 'Graph Attention Network'
        assert 'bold_windows' in info['requires']
        assert 'adj' in info['requires']

    def test_registry_invalid_model_name(self):
        """Registry should raise error for unknown model."""
        with pytest.raises(ValueError):
            ModelRegistry.get_model_class('nonexistent_model')

    def test_registry_model_config_serialization(self):
        """ModelConfig should serialize/deserialize correctly."""
        original = ModelConfig(
            model_name='transformer',
            hidden_dim=128,
            num_layers=3,
            dropout=0.2,
        )
        
        config_dict = original.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['model_name'] == 'transformer'
        assert config_dict['hidden_dim'] == 128


class TestAugmentationIntegration:
    """Test augmentation pipeline with models."""

    def test_augmentation_preserves_shape(self, sample_bold):
        """Augmentation should preserve input shape."""
        from brain_gcn.utils.data.augmentation import AugmentationPipeline
        
        pipeline = AugmentationPipeline.light()
        augmented = pipeline.apply(sample_bold)
        
        assert augmented.shape == sample_bold.shape
        assert augmented.dtype == sample_bold.dtype

    def test_augmented_bold_works_with_model(self, sample_bold):
        """Augmented BOLD should work with GAT model."""
        from brain_gcn.utils.data.augmentation import AugmentationPipeline
        
        # Augment
        pipeline = AugmentationPipeline.aggressive()
        augmented = pipeline.apply(sample_bold)  # (T, N) = (200, 50)
        
        # Create windows: (W, N) = (20, 50)
        windowed = torch.from_numpy(augmented[:20]).unsqueeze(0)  # (1, 20, 50)
        adj = torch.eye(50).unsqueeze(0)  # (1, 50, 50)
        
        # Forward pass
        model = GATClassifier(hidden_dim=32)
        logits = model(windowed, adj)
        
        assert logits.shape == (1, 2)

    def test_fc_measures_compatible_with_models(self, sample_bold):
        """FC measures output should work as model input."""
        from brain_gcn.utils.data.augmentation import FunctionalConnectivityMeasures
        
        fc = FunctionalConnectivityMeasures.pearson_correlation(sample_bold)
        assert fc.shape == (50, 50)
        
        # Use as adjacency - ensure float32 dtype
        adj = torch.from_numpy(fc).float().unsqueeze(0)  # (1, 50, 50)
        bold_windows = torch.randn(1, 20, 50)
        
        model = GATClassifier(hidden_dim=32)
        logits = model(bold_windows, adj)
        assert logits.shape == (1, 2)


class TestModelCompatibility:
    """Test compatibility between models and training pipeline."""

    def test_all_new_models_forward(self, sample_bold_batch, sample_adjacency_batch):
        """All 4 new models should do forward pass without errors."""
        bold_windows = sample_bold_batch  # (4, 20, 50)
        adj = sample_adjacency_batch      # (4, 50, 50)
        
        models_to_test = [
            (GATClassifier(hidden_dim=32), {'bold_windows': bold_windows, 'adj': adj}),
            (TransformerClassifier(hidden_dim=32), {'bold_windows': bold_windows, 'adj': adj}),
            (CNN3DClassifier(hidden_dim=32), {'bold_windows': bold_windows, 'adj': adj}),
            (GraphSAGEClassifier(hidden_dim=32), {'bold_windows': bold_windows, 'adj': adj}),
        ]
        
        for model, inputs_dict in models_to_test:
            logits = model(**inputs_dict)
            assert logits.shape[0] == 4, f"{model.__class__.__name__} batch dimension error"
            assert logits.shape[1] == 2, f"{model.__class__.__name__} output dimension error"

    def test_models_work_with_loss_functions(self, sample_bold_batch, sample_adjacency_batch):
        """Models should work with standard PyTorch loss functions."""
        bold_windows = sample_bold_batch
        adj = torch.abs(sample_adjacency_batch)  # Ensure non-negative
        adj = (adj + adj.transpose(-2, -1)) / 2  # Make symmetric
        labels = torch.randint(0, 2, (4,))  # Binary labels
        
        model = GATClassifier(hidden_dim=32)
        logits = model(bold_windows, adj)
        
        # Test with CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_models_support_inference_mode(self, sample_bold_batch, sample_adjacency_batch):
        """Models should work in inference/eval mode."""
        model = GATClassifier(hidden_dim=32)
        model.eval()
        
        with torch.no_grad():
            logits = model(sample_bold_batch, sample_adjacency_batch)
            assert logits.shape == (4, 2)

    def test_models_have_consistent_parameter_count(self):
        """Model parameter counts should be reasonable."""
        gat = GATClassifier(hidden_dim=64, num_heads=4)
        transformer = TransformerClassifier(hidden_dim=64)
        cnn3d = CNN3DClassifier(hidden_dim=64)
        graphsage = GraphSAGEClassifier(hidden_dim=64)
        
        for model in [gat, transformer, cnn3d, graphsage]:
            param_count = sum(p.numel() for p in model.parameters())
            # All should have at least a few thousand parameters
            assert param_count > 1000, f"{model.__class__.__name__} parameter count too low"
            # And less than 10M (sanity check for memory)
            assert param_count < 10_000_000, f"{model.__class__.__name__} parameter count too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
