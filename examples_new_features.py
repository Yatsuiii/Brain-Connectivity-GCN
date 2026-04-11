"""
Examples demonstrating new models, data pipeline, and visualization features.

Run: python examples_new_features.py
"""

import numpy as np

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

    print("\n[*] All Available Models:")
    ModelRegistry.print_registry()

    print("\n[*] Graph Attention Network Details:")
    gat_info = ModelRegistry.get_model_info('gat')
    for key, val in gat_info.items():
        print(f"  {key:20s}: {val}")

    print("\n[*] Building Transformer Model:")
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

    T, N = 200, 100
    bold = np.random.randn(T, N).astype(np.float32)

    print(f"\nOriginal BOLD shape: {bold.shape}")
    print("\n[*] Individual Augmentations:")

    noisy = BoldAugmentation.gaussian_noise(bold, std=0.01)
    print(f"  [OK] Gaussian noise")

    jittered = BoldAugmentation.temporal_jitter(bold, jitter_std=0.5)
    print(f"  [OK] Temporal jitter")

    dropped = BoldAugmentation.roi_dropout(bold, dropout_rate=0.1)
    print(f"  [OK] ROI dropout")

    scaled = BoldAugmentation.amplitude_scaling(bold, scale_range=(0.8, 1.2))
    print(f"  [OK] Amplitude scaling")

    print("\n[*] Augmentation Pipelines:")

    light = AugmentationPipeline.light()
    print(f"  [OK] Light: {[a[0] for a in light.augmentations]}")

    moderate = AugmentationPipeline.moderate()
    print(f"  [OK] Moderate: {[a[0] for a in moderate.augmentations]}")

    aggressive = AugmentationPipeline.aggressive()
    print(f"  [OK] Aggressive: {[a[0] for a in aggressive.augmentations]}")


# ===========================================================================
# 3. FUNCTIONAL CONNECTIVITY MEASURES
# ===========================================================================
def example_3_fc_measures():
    """Example 3: Multiple FC measures."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Functional Connectivity Measures")
    print("=" * 80)

    from brain_gcn.utils.data.augmentation import FunctionalConnectivityMeasures

    T, N = 200, 50
    bold = np.random.randn(T, N).astype(np.float32)

    print(f"\nBOLD data: {T} timepoints x {N} ROIs")
    print("\n[*] Computing Different FC Measures:")

    fc_pearson = FunctionalConnectivityMeasures.pearson_correlation(bold)
    print(f"  [OK] Pearson: {fc_pearson.shape}, mean {fc_pearson.mean():.4f}")

    fc_partial = FunctionalConnectivityMeasures.partial_correlation(bold)
    print(f"  [OK] Partial: {fc_partial.shape}, mean {fc_partial.mean():.4f}")

    fc_mi = FunctionalConnectivityMeasures.mutual_information(bold, bins=10)
    print(f"  [OK] MI: {fc_mi.shape}, mean {fc_mi.mean():.4f}")

    fc_coh = FunctionalConnectivityMeasures.coherence(bold, freq_range=(0.01, 0.1))
    print(f"  [OK] Coherence: {fc_coh.shape}, mean {fc_coh.mean():.4f}")


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
    motion = np.random.randn(T, 6) * 0.2

    print(f"\nBOLD data: {T} timepoints x {N} ROIs")
    print(f"Motion params: {T} x 6")
    print("\n[*] Preprocessing steps:")

    bold_filtered = SignalPreprocessing.bandpass_filter(
        bold, freq_range=(0.01, 0.1), fs=0.5
    )
    print(f"  [OK] Bandpass filter")

    bold_clean = SignalPreprocessing.motion_scrubbing(
        bold, motion=motion, threshold=0.5
    )
    print(f"  [OK] Motion scrubbing")

    bold_ica = SignalPreprocessing.ica_denoise(bold, n_components=20)
    print(f"  [OK] ICA denoising (20 components)")


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

    print("\n[*] Available Visualization Classes:")

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
            print(f"    - {method}")


# ===========================================================================
# 6. MODEL COMPARISON WORKFLOW
# ===========================================================================
def example_6_model_comparison_workflow():
    """Example 6: Typical workflow for model comparison."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Model Comparison Workflow")
    print("=" * 80)

    print("""
[*] Compare All 8 Models:

Step 1: Run all models
  python -m brain_gcn.experiments \\
    --models graph_temporal gcn gru fc_mlp gat transformer cnn3d graphsage \\
    --max_epochs 50 \\
    --results_csv results/all_models.csv

Step 2: Analyze results
  python -c "
import pandas as pd
df = pd.read_csv('results/all_models.csv')
best = df.loc[df['test_auc'].idxmax()]
print(f'Best model: {best[\"model_name\"]} (AUC: {best[\"test_auc\"]:.3f})')
  "

Step 3: Train best model
  python -m brain_gcn.main \\
    --model_name [BEST_MODEL] \\
    --max_epochs 200

Step 4: Extended evaluation
  python -m brain_gcn.eval_cli \\
    --eval_plot_roc \\
    --eval_plot_pr \\
    --eval_bootstrap_ci

Step 5: Cross-validation
  python -m brain_gcn.cv_cli \\
    --model_name [BEST_MODEL] \\
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
[*] Create Custom Pipeline:

from brain_gcn.utils.data.augmentation import AugmentationPipeline

# Robust augmentation for small dataset
robust_aug = AugmentationPipeline([
    ('gaussian_noise', {'std': 0.015}),
    ('temporal_jitter', {'jitter_std': 0.5}),
    ('roi_dropout', {'dropout_rate': 0.1}),
    ('amplitude_scaling', {'scale_range': (0.9, 1.1)}),
])

# Apply in training
bold_augmented = robust_aug.apply(bold_signal)

# Or integrate with Lightning:
class CustomTask(ClassificationTask):
    def __init__(self, *args, augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation or AugmentationPipeline.light()
    
    def training_step(self, batch, batch_idx):
        bold, adj, labels = batch
        bold = self.augmentation.apply(bold.cpu().numpy())
        return self._step((bold, adj, labels), "train")
    """)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    try:
        example_1_model_registry()
    except Exception as e:
        print(f"  [SKIP] Registry: {type(e).__name__}")

    try:
        example_2_augmentation()
    except Exception as e:
        print(f"  [SKIP] Augmentation: {type(e).__name__}")

    try:
        example_3_fc_measures()
    except Exception as e:
        print(f"  [SKIP] FC Measures: {type(e).__name__}")

    try:
        example_4_preprocessing()
    except Exception as e:
        print(f"  [SKIP] Preprocessing: {type(e).__name__}")

    try:
        example_5_visualization()
    except Exception as e:
        print(f"  [SKIP] Visualization: {type(e).__name__}")

    example_6_model_comparison_workflow()
    example_7_custom_pipeline()

    print("\n" + "=" * 80)
    print("[OK] EXAMPLES COMPLETE")
    print("=" * 80)
    print("""
[*] For more information:
  - Models & Data Pipeline: MODELS_AND_PIPELINE.md
  - Experiments & Evaluation: EXPERIMENTS.md
  - Project Overview: README.md

[*] Quick Start:
  # Train GAT model
  python -m brain_gcn.main --model_name gat --max_epochs 50

  # Compare all 8 models
  python -m brain_gcn.experiments \\
    --models graph_temporal gcn gru fc_mlp gat transformer cnn3d graphsage

  # Hyperparameter search
  python -m brain_gcn.hpo_cli --hpo_n_trials 20

  # Validation with cross-validation
  python -m brain_gcn.cv_cli --cv_n_splits 5
    """)
