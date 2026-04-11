"""
Model registry for centralized model access and configuration.

Simplifies model loading, configuration, and comparison.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

import torch
from torch import nn

log = logging.getLogger(__name__)


# Import all models
def _lazy_import_models():
    """Lazy import to avoid circular dependencies."""
    from brain_gcn.models.brain_gcn import BrainGCNClassifier, GraphOnlyClassifier, TemporalGRUClassifier, ConnectivityMLPClassifier
    from brain_gcn.models.advanced_models import (
        GATClassifier,
        TransformerClassifier,
        CNN3DClassifier,
        GraphSAGEClassifier,
    )
    return {
        # Original models
        'graph_temporal': BrainGCNClassifier,
        'gcn': GraphOnlyClassifier,
        'gru': TemporalGRUClassifier,
        'fc_mlp': ConnectivityMLPClassifier,
        
        # New models
        'gat': GATClassifier,
        'transformer': TransformerClassifier,
        'cnn3d': CNN3DClassifier,
        'graphsage': GraphSAGEClassifier,
    }


class ModelConfig:
    """Configuration for model instantiation."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        num_heads: int = 4,
        num_layers: int = 2,
        readout: str = "attention",
        drop_edge_p: float = 0.1,
        **kwargs
    ):
        """
        Parameters
        ----------
        model_name : str
            Model identifier (must be in registry)
        hidden_dim : int
            Hidden dimension size
        dropout : float
            Dropout probability
        num_heads : int
            Number of attention heads (for GAT, Transformer)
        num_layers : int
            Number of layers (for Transformer)
        readout : str
            Readout method ("attention" or "mean")
        drop_edge_p : float
            Edge dropout probability (for GCN-based models)
        **kwargs
            Additional arguments
        """
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.readout = readout
        self.drop_edge_p = drop_edge_p
        self.kwargs = kwargs

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'readout': self.readout,
            'drop_edge_p': self.drop_edge_p,
            **self.kwargs
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> ModelConfig:
        """Load configuration from dictionary."""
        model_name = config_dict.pop('model_name')
        return cls(model_name, **config_dict)


class ModelRegistry:
    """Central registry for all available models."""

    _models = None
    _configs = {
        'graph_temporal': {
            'display_name': 'Graph-Temporal GCN',
            'description': 'Graph projection per window + GRU temporal encoder',
            'requires': ['bold_windows', 'adj'],
            'parameters': ['hidden_dim', 'dropout', 'readout', 'drop_edge_p'],
        },
        'gcn': {
            'display_name': 'Graph-Only (GCN)',
            'description': 'GCN baseline over ROI average signals',
            'requires': ['bold_windows', 'adj'],
            'parameters': ['hidden_dim', 'dropout', 'drop_edge_p'],
        },
        'gru': {
            'display_name': 'Temporal-Only (GRU)',
            'description': 'GRU baseline without graph structure',
            'requires': ['bold_windows'],
            'parameters': ['hidden_dim', 'dropout'],
        },
        'fc_mlp': {
            'display_name': 'Connectivity MLP',
            'description': 'Static FC adjacency MLP (requires --no-use_population_adj)',
            'requires': ['adj'],
            'parameters': ['hidden_dim', 'dropout'],
        },
        'gat': {
            'display_name': 'Graph Attention Network',
            'description': 'Multi-head graph attention mechanism',
            'requires': ['bold_windows', 'adj'],
            'parameters': ['hidden_dim', 'dropout', 'num_heads'],
        },
        'transformer': {
            'display_name': 'Transformer Encoder',
            'description': 'Transformer-based temporal encoder',
            'requires': ['bold_windows'],
            'parameters': ['hidden_dim', 'dropout', 'num_heads', 'num_layers'],
        },
        'cnn3d': {
            'display_name': '3D-CNN',
            'description': '3D convolution for spatiotemporal features',
            'requires': ['bold_windows', 'fc_windows'],
            'parameters': ['hidden_dim', 'dropout'],
        },
        'graphsage': {
            'display_name': 'GraphSAGE',
            'description': 'Sampling and aggregating graph convolution',
            'requires': ['bold_windows', 'adj'],
            'parameters': ['hidden_dim', 'dropout'],
        },
    }

    @classmethod
    def get_models(cls) -> dict[str, type]:
        """Get all available models."""
        if cls._models is None:
            cls._models = _lazy_import_models()
        return cls._models

    @classmethod
    def get_model_class(cls, model_name: str) -> type:
        """Get model class by name."""
        models = cls.get_models()
        if model_name not in models:
            available = ', '.join(models.keys())
            raise ValueError(
                f"Unknown model: {model_name}\nAvailable: {available}"
            )
        return models[model_name]

    @classmethod
    def build_model(
        cls,
        config: ModelConfig,
        **override_kwargs
    ) -> nn.Module:
        """Build model instance from config.
        
        Parameters
        ----------
        config : ModelConfig
            Model configuration
        **override_kwargs
            Override config parameters
        
        Returns
        -------
        nn.Module
            Instantiated model
        """
        model_class = cls.get_model_class(config.model_name)
        
        # Prepare arguments
        kwargs = {
            'hidden_dim': config.hidden_dim,
            'dropout': config.dropout,
        }
        
        # Add model-specific parameters
        if config.model_name in ['graph_temporal', 'gcn', 'graphsage']:
            kwargs['drop_edge_p'] = config.drop_edge_p
        
        if config.model_name == 'graph_temporal':
            kwargs['readout'] = config.readout
        
        if config.model_name in ['gat', 'transformer']:
            kwargs['num_heads'] = config.num_heads
        
        if config.model_name == 'transformer':
            kwargs['num_layers'] = config.num_layers
        
        # Apply overrides
        kwargs.update(override_kwargs)
        
        # Remove unsupported kwargs
        model_class_init = model_class.__init__
        import inspect
        sig = inspect.signature(model_class_init)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        log.info(f"Building {config.model_name} with {valid_kwargs}")
        return model_class(**valid_kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available models."""
        return list(cls._configs.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a model.
        
        Parameters
        ----------
        model_name : str
            Model name
        
        Returns
        -------
        dict
            Model metadata
        """
        if model_name not in cls._configs:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._configs[model_name]

    @classmethod
    def print_registry(cls) -> None:
        """Print all models and their descriptions."""
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)
        
        for model_name in cls.list_models():
            info = cls.get_model_info(model_name)
            print(f"\n{model_name:15} | {info['display_name']}")
            print(f"{'':15} | {info['description']}")
            print(f"{'':15} | Requires: {', '.join(info['requires'])}")


def add_model_choice_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add model choice argument to parser.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    
    Returns
    -------
    argparse.ArgumentParser
        Updated parser
    """
    available_models = ModelRegistry.list_models()
    
    parser.add_argument(
        '--model_name',
        type=str,
        choices=available_models,
        default='graph_temporal',
        help=f"Model architecture. Available: {', '.join(available_models)}",
    )
    
    parser.add_argument(
        '--num_heads',
        type=int,
        default=4,
        help="Number of attention heads (for GAT, Transformer)",
    )
    
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help="Number of layers (for Transformer)",
    )
    
    return parser


if __name__ == "__main__":
    # Print all available models
    ModelRegistry.print_registry()
