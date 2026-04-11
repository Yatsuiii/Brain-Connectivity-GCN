"""
Examples demonstrating new models, data pipeline, and visualization features.

Run: python examples_new_features.py
"""

import numpy as np
from pathlib import Path

print("\n" + "=" * 80)
print("BRAIN-CONNECTIVITY-GCN: NEW FEATURES SHOWCASE")
print("=" * 80)


# ===========================================================================
# 1. MODEL REGISTRY & COMPARISON
# ===========================================================================
def example_1_model_registry():
    """Example 1: Explore all available models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Model Registry - 8 Available Models")
    print("=" * 80)

    from brain_gcn.models import ModelRegistry, ModelConfig

    # Print all available models
    print("\n📋 All Available Models:")
    ModelRegistry.print_registry()

    # Get specific model info
    print("\n📊 Graph Attention Network Details:")
    gat_info = ModelRegistry.get_model_info('gat')
    for key, val in gat_info.items():
        print(f"  {key:20s}: {val}")

    # Build model from config
    print("\n🔧 Building Transformer Model:")
    config = ModelConfig(
        model_name='transformer',
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.3,
    )
    print(f"  Config: {config.to_dict()}")


# ===========================================================================
# 2. DATA AUGMENTATION
# ===========================================================================
def example_2_augmentation():
    """Example 2: Data augmentation strategies."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Data Augmentation Strategies")
    print("=" * 80)

    from brain_gcn.utils.data.augmentation import (
        BoldAugmentation,
        AugmentationPipeline,
    )

    # Create synthetic BOLD data
    T, N = 200, 100  # 200 timepoints, 100 ROIs
    bold = np.random.randn(T, N).astype(np.float32)

    print(f"\nOriginal BOLD shape: {bold.shape}")

    # Example 1: Individual augmentations
    print("\n📌 Individual Augmentations:")

    noisy = BoldAugmentation.gaussian_noise(bold, std=0.01)
    print(f"  ✓ Gaussian noise applied")

    jittered = BoldAugmentation.temporal_jitter(bold, jitter_std=0.5)
    print(f"  ✓ Temporal jitter applied")

    dropped = BoldAugmentation.roi_dropout(bold, dropout_rate=0.1)
    print(f"  ✓ ROI dropout applied")

    scaled = BoldAugmentation.amplitude_scaling(bold, scale_range=(0.8, 1.2))
    print(f"  ✓ Amplitude scaling applied")

    # Example 2: Preset pipelines
    print("\n🔀 Augmentation Pipelines:")

    light = AugmentationPipeline.light()
    print(f"  ✓ Light pipeline: {[a[0] for a in light.augmentations]}")

    moderate = AugmentationPipeline.moderate()
    print(f"  ✓ Moderate pipeline: {[a[0] for a in moderate.augmentations]}")

    aggressive = AugmentationPipeline.aggressive()
    print(f"  ✓ Aggressive pipeline: {[a[0] for a in aggressive.augmentations]}")

    # Example 3: Custom pipeline
    print("\n⚙️  Custom Pipeline:")
    custom = AugmentationPipeline([
        ('gaussian_noise', {'std': 0.02}),
        ('roi_dropout', {'dropout_rate': 0.15}),
        ('amplitude_scaling', {'scale_range': (0.85, 1.15)}),
    ])
    augmented = custom.apply(bold)
    print(f"  ✓ Applied custom pipeline: {augmented.shape}")


# ===========================================================================
# 3. FUNCTIONAL CONNECTIVITY MEASURES
# ===========================================================================
def example_3_fc_measures():
    """Example 3: Multiple FC measures."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Functional Connectivity Measures")
    print("=" * 80)

    from brain_gcn.utils.data.augmentation import FunctionalConnectivityMeasures

    # Synthetic data
    T, N = 200, 50
    bold = np.random.randn(T, N).astype(np.float32)

    print(f"\nBOLD data: {T} timepoints × {N} ROIs")

    # Pearson correlation
    print("\n📊 Computing Different FC Measures:")

    fc_pearson = FunctionalConnectivityMeasures.pearson_correlation(bold)
    print(f"  ✓ Pearson correlation: shape {fc_pearson.shape}, mean {fc_pearson.mean():.4f}")

    fc_partial = FunctionalConnectivityMeasures.partial_correlation(bold)
    print(f"  ✓ Partial correlation: shape {fc_partial.shape}, mean {fc_partial.mean():.4f}")

    fc_mi = FunctionalConnectivityMeasures.mutual_information(bold, bins=10)
    print(f"  ✓ Mutual information: shape {fc_mi.shape}, mean {fc_mi.mean():.4f}")

    fc_coh = FunctionalConnectivityMeasures.coherence(bold, freq_range=(0.01, 0.1))
    print(f"  ✓ Frequency coherence: shape {fc_coh.shape}, mean {fc_coh.mean():.4f}")


# ===========================================================================
# 4. SIGNAL PREPROCESSING
# ===========================================================================
def example_4_preprocessing():
    """Example 4: Advanced signal preprocessing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Signal Preprocessing")
    print("=" * 80)

    from brain_gcn.utils.data.augmentation import SignalPreprocessing

    T, N = 300, 50
    bold = np.random.randn(T, N).astype(np.float32)
    motion = np.random.randn(T, 6) * 0.2  # Random motion params

    print(f"\nBOLD data: {T} timepoints × {N} ROIs")
    print(f"Motion params: {T} timepoints × 6 parameters")

    # Bandpass filtering
    print("\n🔧 Preprocessing steps:")

    bold_filtered = SignalPreprocessing.bandpass_filter(
        bold,
        freq_range=(0.01, 0.1),
        fs=0.5
    )
    print(f"  ✓ Bandpass filtered {bold.shape} → {bold_filtered.shape}")

    # Motion scrubbing
    bold_clean = SignalPreprocessing.motion_scrubbing(
        bold,
        motion=motion,
        threshold=0.5
    )
    print(f"  ✓ Motion scrubbing applied")

    # ICA denoising
    bold_ica = SignalPreprocessing.ica_denoise(bold, n_components=20)
    print(f"  ✓ ICA denoising: kept 20 components")


# ===========================================================================
# 5. VISUALIZATION EXAMPLES
# ===========================================================================
def example_5_visualization():
    """Example 5: Visualization suite."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Visualization Suite")
    print("=" * 80)

    from brain_gcn.utils.visualization import (
        BrainConnectivityVisualizer,
        ModelAnalyzer,
        TrainingAnalyzer,
        AttentionVisualizer,
        StatisticalVisualizer,
    )

    print("\n📊 Available Visualization Functions:")

    visualizers = [
        ("BrainConnectivityVisualizer", [
            "plot_connectivity_matrix",
            "plot_connectivity_comparison",
            "plot_dynamic_connectivity",
        ]),
        ("ModelAnalyzer", [
            "plot_model_comparison",
            "plot_confusion_matrix",
        ]),
        ("TrainingAnalyzer", [
            "plot_training_curves",
            "plot_learning_rate_schedule",
        ]),
        ("AttentionVisualizer", [
            "plot_roi_attention",
        ]),
        ("StatisticalVisualizer", [
            "plot_group_comparison",
        ]),
    ]

    for viz_class, methods in visualizers:
        print(f"\n  {viz_class}:")
        for method in methods:
            print(f"    • {method}")

    print("\n💡 Example: Model Comparison Plot")
    print("  from brain_gcn.utils.visualization import ModelAnalyzer")
    print("  results = {")
    print("      'graph_temporal': {'test_auc': 0.856},")
    print("      'gat': {'test_auc': 0.842},")
    print("      'transformer': {'test_auc': 0.834},")
    print("  }")
    print("  ModelAnalyzer.plot_model_comparison(results, output_path='models.png')")


# ===========================================================================
# 6. MODEL COMPARISON WORKFLOW
# ===========================================================================
def example_6_model_comparison_workflow():
    """Example 6: Typical workflow for model comparison."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Model Comparison Workflow")
    print("=" * 80)

    print("""
📋 Typical Workflow to Compare All 8 Models:

Step 1: Run experiment on all models
  python -m brain_gcn.experiments \\
    --models graph_temporal gcn gru fc_mlp gat transformer cnn3d graphsage \\
    --max_epochs 50 \\
    --results_csv results/all_models.csv

Step 2: Visualize comparison
  python -c "
from brain_gcn.utils.visualization import ModelAnalyzer
import pandas as pd

df = pd.read_csv('results/all_models.csv')
results = {row['model_name']: row.to_dict() 
           for _, row in df.iterrows()}
ModelAnalyzer.plot_model_comparison(results, output_path='comparison.png')
  "

Step 3: Find best model
  best_model = df.loc[df['test_auc'].idxmax(), 'model_name']
  print(f"Best model: {best_model}")

Step 4: Train best model with extended evaluation
  python -m brain_gcn.main \\
    --model_name {best_model} \\
    --max_epochs 200 \\
    --test
  
  python -m brain_gcn.eval_cli \\
    --eval_checkpoint checkpoints/{best_model}/best.ckpt \\
    --eval_plot_roc \\
    --eval_plot_pr \\
    --eval_bootstrap_ci

Step 5: Validate with cross-validation
  python -m brain_gcn.cv_cli \\
    --model_name {best_model} \\
    --cv_n_splits 5
    """)


# ===========================================================================
# 7. CUSTOM AUGMENTATION PIPELINE
# ===========================================================================
def example_7_custom_pipeline():
    """Example 7: Creating custom augmentation pipeline."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Augmentation Pipeline")
    print("=" * 80)

    print("""
📌 Create a Custom Pipeline for Your Use Case:

from brain_gcn.utils.data.augmentation import AugmentationPipeline

# Example: Robust augmentation for small dataset
robust_aug = AugmentationPipeline([
    ('gaussian_noise', {'std': 0.015}),
    ('temporal_jitter', {'jitter_std': 0.5}),
    ('roi_dropout', {'dropout_rate': 0.1}),
    ('amplitude_scaling', {'scale_range': (0.9, 1.1)}),
])

# Use in training
bold_augmented = robust_aug.apply(bold_signal)

# Or integrate with Lightning
class CustomTask(ClassificationTask):
    def __init__(self, *args, augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation or AugmentationPipeline.light()
    
    def training_step(self, batch, batch_idx):
        bold_windows, adj, labels = batch
        # Apply augmentation
        bold_aug = self.augmentation.apply(bold_windows.cpu().numpy())
        bold_windows = torch.tensor(bold_aug)
        return self._step((bold_windows, adj, labels), "train")
    """)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    try:
        example_1_model_registry()
    except Exception as e:
        print(f"  ⚠️  Skipped (import): {e}")

    try:
        example_2_augmentation()
    except Exception as e:
        print(f"  ⚠️  Skipped: {e}")

    try:
        example_3_fc_measures()
    except Exception as e:
        print(f"  ⚠️  Skipped: {e}")

    try:
        example_4_preprocessing()
    except Exception as e:
        print(f"  ⚠️  Skipped: {e}")

    try:
        example_5_visualization()
    except Exception as e:
        print(f"  ⚠️  Skipped: {e}")

    example_6_model_comparison_workflow()
    example_7_custom_pipeline()

    print("\n" + "=" * 80)
    print("✅ EXAMPLES COMPLETE")
    print("=" * 80)
    print("""
📚 For more information:
  • Models & Pipeline: See MODELS_AND_PIPELINE.md
  • Experiments: See EXPERIMENTS.md
  • Main README: See README.md

🚀 Quick Start:
  # Train with GAT model
  python -m brain_gcn.main --model_name gat --max_epochs 50

  # Compare all 8 models
  python -m brain_gcn.experiments \\
    --models graph_temporal gcn gru fc_mlp gat transformer cnn3d graphsage

  # Hyperparameter search
  python -m brain_gcn.hpo_cli --hpo_n_trials 20
    """)
