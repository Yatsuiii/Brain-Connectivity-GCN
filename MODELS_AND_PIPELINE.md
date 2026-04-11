# Advanced Models & Enhanced Pipeline

## New Capabilities (Phase 2 Expansion)

This document describes the major enhancements added to the Brain-Connectivity-GCN project:

1. **4 New Model Architectures** beyond the original 4 baselines
2. **Advanced Data Pipeline** with augmentation, preprocessing, measures
3. **Comprehensive Visualization Suite** for analysis and reporting

---

## 1. New Model Architectures

### Available Models (8 Total)

| Model | Class | Type | Requires | Key Feature |
|-------|-------|------|----------|-------------|
| `graph_temporal` | BrainGCNClassifier | GCN + RNN | bold, adj | 2-layer GCN + GRU temporal |
| `gcn` | GraphOnlyClassifier | GCN | bold, adj | GCN baseline |
| `gru` | TemporalGRUClassifier | RNN | bold | Temporal baseline |
| `fc_mlp` | ConnectivityMLPClassifier | MLP | adj | Static connectivity MLP |
| `gat` | GATClassifier | Graph Attention | bold, adj | **Multi-head attention** ✨ |
| `transformer` | TransformerClassifier | Transformer | bold | **Self-attention temporal** ✨ |
| `cnn3d` | CNN3DClassifier | 3D-CNN | bold, adj | **Spatiotemporal convolution** ✨ |
| `graphsage` | GraphSAGEClassifier | GraphSAGE | bold, adj | **Sampling & aggregating** ✨ |

**New models marked with ✨**

### Model Details

#### Graph Attention Networks (GAT)
```python
from brain_gcn.models import GATClassifier

# Multi-head graph attention
model = GATClassifier(
    hidden_dim=64,
    num_heads=4,        # Parallel attention heads
    dropout=0.5
)

# Run: python -m brain_gcn.main --model_name gat --num_heads 8
```

**Architecture:**
- Learns separate attention weights for each neighbor
- Multiple attention heads → different relationship types
- Better at discovering complex patterns

**When to use:**
- Small brain networks (< 300 ROIs) where attention patterns matter
- When you want interpretable edge importance

---

#### Transformer Encoder
```python
from brain_gcn.models import TransformerClassifier

# Self-attention on temporal sequences
model = TransformerClassifier(
    hidden_dim=64,
    num_heads=4,
    num_layers=2,       # Number of transformer blocks
    dropout=0.5
)

# Run: python -m brain_gcn.main --model_name transformer --num_layers 3
```

**Architecture:**
- Self-attention on sliding windows (no graph)
- Captures temporal dependencies directly
- Multiple layers of transformer blocks

**When to use:**
- Focus on temporal dynamics over spatial structure
- When graph structure is less important
- For sequence-to-sequence modeling

---

#### 3D-CNN for Spatiotemporal Analysis
```python
from brain_gcn.models import CNN3DClassifier

# 3D convolutions on connectivity dynamics
model = CNN3DClassifier(
    hidden_dim=64,
    dropout=0.5
)

# Run: python -m brain_gcn.main --model_name cnn3d
```

**Architecture:**
- Treats FC matrices as spatial volumes
- Convolutional filters across space + time
- Learns spatiotemporal feature maps

**When to use:**
- Connectivity patterns that evolve over time
- When local connectivity clusters matter
- As complement to graph methods

---

#### GraphSAGE (Graph Sampling & Aggregating)
```python
from brain_gcn.models import GraphSAGEClassifier

# Inductive graph learning via sampling
model = GraphSAGEClassifier(
    hidden_dim=64,
    dropout=0.5
)

# Run: python -m brain_gcn.main --model_name graphsage
```

**Architecture:**
- Mean aggregation of neighboring ROI features
- Two-layer sage network
- Generalizes to unseen nodes

**When to use:**
- Inductive settings or new subjects
- Mean neighborhood aggregation works well
- Simpler alternative to attention

---

### Model Registry & Comparison

```python
from brain_gcn.models import ModelRegistry

# List all models
ModelRegistry.print_registry()

# Get model info
info = ModelRegistry.get_model_info('gat')
# {'display_name': 'Graph Attention Network', ...}

# Build model from config
from brain_gcn.models import ModelConfig

config = ModelConfig(
    model_name='transformer',
    hidden_dim=128,
    num_heads=8,
    num_layers=3,
)
model = ModelRegistry.build_model(config)
```

---

## 2. Advanced Data Pipeline

### Augmentation Strategies

#### Light Augmentation (Default for stability)
```python
from brain_gcn.utils.data.augmentation import AugmentationPipeline

pipeline = AugmentationPipeline.light()
# → Gaussian noise (std=0.005)
```

#### Moderate Augmentation
```python
pipeline = AugmentationPipeline.moderate()
# → Gaussian noise + temporal jitter
```

#### Aggressive Augmentation
```python
pipeline = AugmentationPipeline.aggressive()
# → Noise + jitter + ROI dropout + amplitude scaling
```

#### Custom Pipeline
```python
pipeline = AugmentationPipeline([
    ('gaussian_noise', {'std': 0.01}),
    ('roi_dropout', {'dropout_rate': 0.15}),
    ('amplitude_scaling', {'scale_range': (0.85, 1.15)}),
])
```

**Available Augmentations:**

| Augmentation | Effect | Use Case |
|--------------|--------|----------|
| `gaussian_noise` | Add white noise | General regularization |
| `temporal_jitter` | Random time shifts per ROI | Temporal robustness |
| `roi_dropout` | Zero-out entire time series | Robustness to missing data |
| `frequency_dropout` | Zero frequency components | Signal noise resistance |
| `time_warping` | Non-linear time warping | Temporal variability |
| `amplitude_scaling` | Rescale per-ROI | Signal magnitude robustness |

---

### Multiple Functional Connectivity Measures

```python
from brain_gcn.utils.data.augmentation import FunctionalConnectivityMeasures

bold = dataset.load_bold_signal()  # (T, N)

# Standard Pearson correlation
fc_pearson = FunctionalConnectivityMeasures.pearson_correlation(bold)

# Partial correlation (direct connections)
fc_partial = FunctionalConnectivityMeasures.partial_correlation(bold)

# Mutual information (non-linear relationships)
fc_mi = FunctionalConnectivityMeasures.mutual_information(bold, bins=10)

# Frequency-domain coherence
fc_coherence = FunctionalConnectivityMeasures.coherence(
    bold,
    freq_range=(0.01, 0.1),  # Low-frequency band
    fs=0.5  # Sampling frequency
)
```

**When to use each:**
- **Pearson**: Standard baseline, interpretable
- **Partial**: Real connections (removes volume conduction)
- **Mutual Info**: Non-linear patterns
- **Coherence**: Phase-locked activity in specific bands

---

### Signal Preprocessing

```python
from brain_gcn.utils.data.augmentation import SignalPreprocessing

# Bandpass filter to remove low-frequency drift and scanner noise
bold_filtered = SignalPreprocessing.bandpass_filter(
    bold,
    freq_range=(0.01, 0.1),  # Typical BOLD band
    fs=0.5
)

# Remove high-motion frames and interpolate
bold_clean = SignalPreprocessing.motion_scrubbing(
    bold,
    motion=motion_parameters,  # (T, 6)
    threshold=0.5  # mm
)

# ICA-based denoising
bold_ica = SignalPreprocessing.ica_denoise(
    bold,
    n_components=20  # Keep 20 ICA components
)
```

---

### Multi-Site Normalization

```python
from brain_gcn.utils.data.augmentation import MultiSiteNormalization

# Normalize across sites to remove batch effects
harmonized = MultiSiteNormalization.harmonization(
    bold_list=[bold_site1, bold_site2, bold_site3],
    sites=['PITT', 'OHSU', 'OLIN']
)
```

---

## 3. Visualization & Analysis Suite

### Brain Connectivity Visualization

```python
from brain_gcn.utils.visualization import BrainConnectivityVisualizer

# Plot connectivity matrix
visualizer = BrainConnectivityVisualizer()

visualizer.plot_connectivity_matrix(
    connectivity_matrix,
    title="Mean ASD Connectivity",
    output_path="asd_connectivity.png"
)

# Compare groups
visualizer.plot_connectivity_comparison(
    conn_asd=asd_mean_connectivity,
    conn_td=td_mean_connectivity,
    output_path="group_comparison.png"
)

# Temporal dynamics
visualizer.plot_dynamic_connectivity(
    fc_windows,  # (W, N, N)
    output_path="dynamic_connectivity.png"
)
```

---

### Model Analysis & Comparison

```python
from brain_gcn.utils.visualization import ModelAnalyzer

# Compare model performance
analyzer = ModelAnalyzer()

results = {
    'graph_temporal': {'test_auc': 0.856, 'test_acc': 0.802},
    'gat': {'test_auc': 0.842, 'test_acc': 0.795},
    'transformer': {'test_auc': 0.834, 'test_acc': 0.788},
}

analyzer.plot_model_comparison(
    results,
    metric='test_auc',
    output_path="model_comparison.png"
)

# Confusion matrix
analyzer.plot_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png")
```

---

### Training Analysis

```python
from brain_gcn.utils.visualization import TrainingAnalyzer

trainer = TrainingAnalyzer()

# Plot loss and metric curves
trainer.plot_training_curves(
    train_loss=[...],
    val_loss=[...],
    train_metric=[...],
    val_metric=[...],
    metric_name='AUC',
    output_path="training_curves.png"
)

# Learning rate schedule
trainer.plot_learning_rate_schedule(
    lrs=[...],
    output_path="lr_schedule.png"
)
```

---

### Attention Visualization

```python
from brain_gcn.utils.visualization import AttentionVisualizer

# Visualize ROI attention weights (e.g., from graph attention models)
visualizer = AttentionVisualizer()

visualizer.plot_roi_attention(
    attention_weights,  # (N,)
    roi_names=roi_names,  # Optional
    output_path="top_rois.png",
    top_k=20
)
```

---

### Statistical Visualization

```python
from brain_gcn.utils.visualization import StatisticalVisualizer

visualizer = StatisticalVisualizer()

# Violin plot of group differences
visualizer.plot_group_comparison(
    asd_values=asd_connectivity_strength,
    td_values=td_connectivity_strength,
    metric_name="Mean Connectivity Strength",
    output_path="group_comparison.png"
)
```

---

## Quick Start: Testing New Models

### 1. Train Single New Model
```bash
python -m brain_gcn.main \
  --model_name gat \
  --num_heads 8 \
  --hidden_dim 128 \
  --dropout 0.5 \
  --max_epochs 50 \
  --test
```

### 2. Compare All 8 Models
```bash
python -m brain_gcn.experiments \
  --models graph_temporal gcn gru fc_mlp gat transformer cnn3d graphsage \
  --max_epochs 50 \
  --results_csv results/all_models_comparison.csv
```

### 3. Ablate Specific Model
```bash
python -m brain_gcn.ablation_cli \
  --model_name gat \
  --num_heads 4
```

### 4. Hyperparameter Search for GAT
```bash
python -m brain_gcn.hpo_cli \
  --model_name gat \
  --hpo_n_trials 30 \
  --hpo_objective test_auc
```

---

## Example Workflow: GAT Evaluation

```bash
# 1. Find best GAT hyperparameters
python -m brain_gcn.hpo_cli \
  --model_name gat \
  --hpo_n_trials 20

# 2. Train final model with best params
python -m brain_gcn.main \
  --model_name gat \
  --num_heads 8 \
  --hidden_dim 128 \
  --dropout 0.3 \
  --lr 0.0005 \
  --max_epochs 200 \
  --test

# 3. Validate with CV
python -m brain_gcn.cv_cli \
  --model_name gat \
  --cv_n_splits 5

# 4. Evaluate with extended metrics
python -m brain_gcn.eval_cli \
  --eval_checkpoint checkpoints/gat/best.ckpt \
  --eval_plot_roc \
  --eval_plot_pr \
  --eval_bootstrap_ci

# 5. Visualize results
python -c "
from brain_gcn.utils.visualization import create_analysis_summary
create_analysis_summary('results/gat_analysis', model_results)
"
```

---

## Dependencies

New dependencies (already included):
```
matplotlib>=3.5.0    # Visualization
seaborn>=0.11.0      # Statistical plots
scikit-learn>=1.0    # ICA, preprocessing
scipy>=1.8.0         # Signal processing
```

---

## Performance Notes

| Model | Speed | Memory | Best For |
|-------|-------|--------|----------|
| graph_temporal | ⚡⚡ Fast | ⚡ Low | Baseline, general use |
| gcn | ⚡⚡ Fast | ⚡ Low | Graph-only baseline |
| gru | ⚡⚡ Fast | ⚡ Low | Temporal baseline |
| fc_mlp | ⚡⚡⚡ Very fast | ⚡ Very low | Lightweight |
| gat | ⚡ Medium | ⚡⚡ Medium | Attention interpretability |
| transformer | ⚡ Medium | ⚡⚡ Medium | Temporal patterns |
| cnn3d | ⚡⚡ Fast | ⚡⚡ Medium | Spatiotemporal |
| graphsage | ⚡⚡ Fast | ⚡ Low | Inductive settings |

---

## Advanced Usage: Custom Augmentation

```python
from brain_gcn.utils.data.augmentation import AugmentationPipeline, BoldAugmentation
from brain_gcn.tasks import ClassificationTask

# Create custom augmentation
custom_aug = AugmentationPipeline([
    ('gaussian_noise', {'std': 0.02}),
    ('frequency_dropout', {'freq_dropout_rate': 0.15}),
    ('time_warping', {'warping_factor': 0.1}),
])

# Extend ClassificationTask to use custom augmentation
class CustomTask(ClassificationTask):
    def training_step(self, batch, batch_idx):
        bold_windows, adj, labels = batch
        # Apply custom augmentation
        bold_windows = custom_aug.apply(bold_windows.cpu().numpy())
        bold_windows = torch.tensor(bold_windows, device=adj.device)
        return self._step((bold_windows, adj, labels), "train")
```

---

## Next Steps

1. **Compare all 8 models** on your dataset
2. **Experiment with augmentation** levels
3. **Visualize attention patterns** for interpretability
4. **Perform ablation studies** on individual components
5. **Export best model** for deployment

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experimental workflows.

