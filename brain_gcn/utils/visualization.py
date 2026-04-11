"""
Comprehensive visualization and analysis suite.

Features:
- Model comparison plots
- Brain connectivity heatmaps
- Training curves and loss landscapes
- Confusion matrices and ROC curves (already in evaluation.py)
- Feature importance and attention maps
- Interactive dashboards (via plotly)
- Statistical group comparisons
- Model ensemble visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brain Connectivity Visualization
# ---------------------------------------------------------------------------

class BrainConnectivityVisualizer:
    """Visualize functional connectivity patterns."""

    @staticmethod
    def plot_connectivity_matrix(
        connectivity: np.ndarray,
        title: str = "Functional Connectivity",
        output_path: str | Path | None = None,
        cmap: str = "coolwarm",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Plot connectivity matrix as heatmap.
        
        Parameters
        ----------
        connectivity : (N, N) array
            Connectivity matrix
        title : str
            Plot title
        output_path : Path, optional
            Save figure
        cmap : str
            Colormap
        vmin, vmax : float
            Color scale limits
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(connectivity, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation", rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()

    @staticmethod
    def plot_connectivity_comparison(
        conn_asd: np.ndarray,
        conn_td: np.ndarray,
        title: str = "Connectivity Comparison (ASD vs TD)",
        output_path: str | Path | None = None,
    ) -> None:
        """Compare group connectivity patterns.
        
        Parameters
        ----------
        conn_asd, conn_td : (N, N) arrays
            Mean connectivity for each group
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        vmax = max(np.abs(conn_asd).max(), np.abs(conn_td).max())
        
        # ASD
        im1 = axes[0].imshow(conn_asd, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[0].set_title("ASD Mean", fontweight='bold')
        axes[0].set_xlabel("ROI")
        axes[0].set_ylabel("ROI")
        plt.colorbar(im1, ax=axes[0])
        
        # TD
        im2 = axes[1].imshow(conn_td, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[1].set_title("TD Mean", fontweight='bold')
        axes[1].set_xlabel("ROI")
        axes[1].set_ylabel("ROI")
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = conn_asd - conn_td
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axes[2].set_title("ASD - TD", fontweight='bold')
        axes[2].set_xlabel("ROI")
        axes[2].set_ylabel("ROI")
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()

    @staticmethod
    def plot_dynamic_connectivity(
        fc_windows: np.ndarray,
        output_path: str | Path | None = None,
    ) -> None:
        """Visualize connectivity dynamics over time.
        
        Takes mean correlation strength per window.
        
        Parameters
        ----------
        fc_windows : (W, N, N) array
            Connectivity per window
        """
        # Compute mean absolute connectivity per window
        strength = np.abs(fc_windows).mean(axis=(1, 2))
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(strength, linewidth=2, color='steelblue')
        ax.fill_between(range(len(strength)), strength, alpha=0.3, color='steelblue')
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Mean Connectivity Strength")
        ax.set_title("Dynamic Functional Connectivity", fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()


# ---------------------------------------------------------------------------
# Model Analysis & Comparison
# ---------------------------------------------------------------------------

class ModelAnalyzer:
    """Analyze and compare model performance."""

    @staticmethod
    def plot_model_comparison(
        results: dict[str, dict],
        metric: str = "test_auc",
        output_path: str | Path | None = None,
    ) -> None:
        """Compare metrics across models.
        
        Parameters
        ----------
        results : dict
            {model_name: {metric: value, ...}, ...}
        metric : str
            Metric to compare
        """
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f"Model Comparison: {metric}", fontweight='bold', fontsize=14)
        ax.set_ylim([0, max(values) * 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """Plot confusion matrix heatmap.
        
        Parameters
        ----------
        y_true, y_pred : (N,) arrays
            True and predicted labels
        labels : list[str]
            Class names (e.g., ["TD", "ASD"])
        """
        if labels is None:
            labels = ["Class 0", "Class 1"]
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        
        ax.set_ylabel("True Label", fontweight='bold')
        ax.set_xlabel("Predicted Label", fontweight='bold')
        ax.set_title("Confusion Matrix", fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()


# ---------------------------------------------------------------------------
# Training Analysis
# ---------------------------------------------------------------------------

class TrainingAnalyzer:
    """Analyze training dynamics."""

    @staticmethod
    def plot_training_curves(
        train_loss: list[float],
        val_loss: list[float],
        train_metric: list[float] | None = None,
        val_metric: list[float] | None = None,
        metric_name: str = "AUC",
        output_path: str | Path | None = None,
    ) -> None:
        """Plot loss and metric curves.
        
        Parameters
        ----------
        train_loss, val_loss : list[float]
            Training/validation loss per epoch
        train_metric, val_metric : list[float], optional
            Training/validation metric per epoch
        metric_name : str
            Name of metric (e.g., "AUC", "Accuracy")
        """
        epochs = range(1, len(train_loss) + 1)
        
        if train_metric is not None and val_metric is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # Loss
        ax1.plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=4)
        ax1.plot(epochs, val_loss, 's-', label='Validation', linewidth=2, markersize=4)
        ax1.set_xlabel("Epoch", fontweight='bold')
        ax1.set_ylabel("Loss", fontweight='bold')
        ax1.set_title("Training Loss", fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Metric
        if train_metric is not None and val_metric is not None:
            ax2.plot(epochs, train_metric, 'o-', label='Train', linewidth=2, markersize=4)
            ax2.plot(epochs, val_metric, 's-', label='Validation', linewidth=2, markersize=4)
            ax2.set_xlabel("Epoch", fontweight='bold')
            ax2.set_ylabel(metric_name, fontweight='bold')
            ax2.set_title(f"Training {metric_name}", fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()

    @staticmethod
    def plot_learning_rate_schedule(
        lrs: list[float],
        output_path: str | Path | None = None,
    ) -> None:
        """Visualize learning rate schedule.
        
        Parameters
        ----------
        lrs : list[float]
            Learning rate per epoch
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(range(1, len(lrs) + 1), lrs, 'o-', linewidth=2, markersize=5)
        ax.set_xlabel("Epoch", fontweight='bold')
        ax.set_ylabel("Learning Rate", fontweight='bold')
        ax.set_title("Learning Rate Schedule", fontweight='bold', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()


# ---------------------------------------------------------------------------
# Attention & Feature Importance
# ---------------------------------------------------------------------------

class AttentionVisualizer:
    """Visualize model attention mechanisms."""

    @staticmethod
    def plot_roi_attention(
        attention_weights: np.ndarray,
        roi_names: list[str] | None = None,
        output_path: str | Path | None = None,
        top_k: int = 20,
    ) -> None:
        """Plot top ROIs by attention weight.
        
        Parameters
        ----------
        attention_weights : (N,) array
            Attention weight per ROI
        roi_names : list[str], optional
            ROI names
        top_k : int
            Number of top ROIs to show
        """
        top_idx = np.argsort(attention_weights)[-top_k:][::-1]
        top_weights = attention_weights[top_idx]
        
        if roi_names is None:
            roi_labels = [f"ROI {i}" for i in top_idx]
        else:
            roi_labels = [roi_names[i] for i in top_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top_weights)), top_weights, color='viridis')
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_weights)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(range(len(top_weights)))
        ax.set_yticklabels(roi_labels, fontsize=10)
        ax.set_xlabel("Attention Weight", fontweight='bold')
        ax.set_title(f"Top {top_k} ROIs by Attention", fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()


# ---------------------------------------------------------------------------
# Statistical Visualization
# ---------------------------------------------------------------------------

class StatisticalVisualizer:
    """Visualize statistical group differences."""

    @staticmethod
    def plot_group_comparison(
        asd_values: np.ndarray,
        td_values: np.ndarray,
        metric_name: str = "Metric",
        output_path: str | Path | None = None,
    ) -> None:
        """Violin plot of group differences.
        
        Parameters
        ----------
        asd_values, td_values : (N,) arrays
            Metric values for each group
        metric_name : str
            Name of metric
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = [td_values, asd_values]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["TD", "ASD"])
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f"Group Comparison: {metric_name}", fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved to {output_path}")
        
        plt.close()


# ---------------------------------------------------------------------------
# Visualization Registry
# ---------------------------------------------------------------------------

class VisualizationRegistry:
    """Registry for all visualization functions."""

    BRAIN_CONNECTIVITY = BrainConnectivityVisualizer
    MODEL_ANALYSIS = ModelAnalyzer
    TRAINING = TrainingAnalyzer
    ATTENTION = AttentionVisualizer
    STATISTICS = StatisticalVisualizer


def create_analysis_summary(
    results_dir: str | Path,
    model_results: dict,
    connectivity_data: dict | None = None,
) -> None:
    """Generate comprehensive analysis summary.
    
    Parameters
    ----------
    results_dir : Path
        Output directory for figures
    model_results : dict
        Dictionary of {model_name: {metric: value}}
    connectivity_data : dict, optional
        {group: connectivity_matrix}
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model comparison
    ModelAnalyzer.plot_model_comparison(
        model_results,
        metric="test_auc",
        output_path=results_dir / "01_model_comparison_auc.png",
    )
    
    # Connectivity comparison if provided
    if connectivity_data and 'asd' in connectivity_data and 'td' in connectivity_data:
        BrainConnectivityVisualizer.plot_connectivity_comparison(
            connectivity_data['asd'],
            connectivity_data['td'],
            output_path=results_dir / "02_connectivity_comparison.png",
        )
    
    log.info(f"Analysis summary saved to {results_dir}")
