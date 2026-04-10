# Experiment Management & Evaluation Enhancements

This document describes the new experiment management and evaluation infrastructure added to Brain-Connectivity-GCN.

## Overview

The project now includes:

1. **Hyperparameter Optimization (HPO)** — automated search using Optuna
2. **Cross-Validation Framework** — K-fold and leave-one-site-out approaches  
3. **Ablation Studies** — systematic component importance analysis
4. **Extended Evaluation Metrics** — comprehensive classification analysis
5. **Experiment Tracking** — metadata logging and result management

---

## 1. Hyperparameter Optimization (HPO)

### Module: `brain_gcn/hpo.py`

Automated hyperparameter search using Optuna's Tree-structured Parzen Estimator (TPE) sampler.

**Search space** includes:
- Model: `hidden_dim` ∈ {32, 64, 128, 256}
- Dropout: `dropout` ∈ [0.0, 0.5]
- Training: `lr` ∈ [1e-5, 1e-2], `weight_decay` ∈ [1e-6, 1e-3], `batch_size` ∈ {8, 16, 32, 64}
- Regularization: `drop_edge_p` ∈ [0.0, 0.3], `bold_noise_std` ∈ [0.0, 0.05]
- Scheduling: `cosine_t0` ∈ {30, 50, 100}, `cosine_t_mult` ∈ {1, 2, 3}

**API:**

```python
from brain_gcn.hpo import HPOConfig, HPOStudy

# Create configuration
config = HPOConfig(
    study_name="brain_gcn_hpo",
    n_trials=20,
    objective_metric="test_auc",
    storage="hpo_studies.db",  # Persistent SQLite storage
)

# Run study
study = HPOStudy(config)
study.create_study()
study.optimize(base_args)

# Retrieve results
print(f"Best AUC: {study.best_value():.4f}")
print(f"Best params: {study.best_params()}")

# Save summary
study.save_summary("results/hpo_summary.json")
```

**CLI:**

```bash
# Run HPO for 20 trials
python -m brain_gcn.hpo_cli \
  --hpo_n_trials 20 \
  --hpo_objective test_auc \
  --hpo_study_name my_study \
  --model_name graph_temporal \
  --max_epochs 50
```

**Key features:**
- Persistent studies (SQLite)
- Median-based pruning for early stopping
- Seeded TPE for reproducibility
- Automatic trial failure handling

---

## 2. Cross-Validation Framework

### Module: `brain_gcn/utils/cross_validation.py`

Implements stratified K-fold and leave-one-site-out cross-validation.

**API:**

```python
from brain_gcn.utils.cross_validation import kfold_cross_validate, CrossValidator

# Run 5-fold cross-validation
cv_results = kfold_cross_validate(
    base_args=args,
    n_splits=5,
    output_dir="results/cv",
)

# Access results
print(cv_results.mean_metrics())
# Output: {'test_auc_mean': 0.85, 'test_auc_std': 0.04, ...}
```

**Leave-one-site-out:**

```python
from brain_gcn.utils.cross_validation import LeaveOneSiteOutValidator

validator = LeaveOneSiteOutValidator()
splits = validator.split(sites_array)  # [(train_idx, test_idx), ...]

for train_idx, test_idx in splits:
    # Train on train_idx, evaluate on test_idx
    pass
```

**Output:**
- `fold_0.pt`, `fold_1.pt`, ... — individual fold results
- `cv_summary.json` — aggregated metrics with mean/std

**CLI:**

```bash
python -m brain_gcn.cv_cli \
  --cv_n_splits 5 \
  --cv_output_dir results/cv \
  --model_name graph_temporal
```

---

## 3. Ablation Studies

### Module: `brain_gcn/ablation.py`

Systematically removes components to measure their contribution.

**Predefined components:**
- `drop_edge` — DropEdge regularization
- `bold_noise` — BOLD signal augmentation
- `graph` — Graph structure (use GRU-only)
- `population_adj` — Population adjacency matrix
- `layer_norm` — Layer normalization

**API:**

```python
from brain_gcn.ablation import AblationStudy

# Create ablation study
study = AblationStudy(
    base_args=args,
    components=["drop_edge", "bold_noise", "graph"],
    output_dir="results/ablations",
)

# Run full study (trains baseline + all ablations)
results = study.run()

# View summary
print(study.summary())

# Save results
study.save_results()  # → ablation_results.json
```

**Output format:**

```json
{
  "results": {
    "baseline": {"test_auc": 0.85, "test_acc": 0.80},
    "drop_edge": {"test_auc": 0.82, "test_acc": 0.78},
    "bold_noise": {"test_auc": 0.83, "test_acc": 0.79}
  },
  "deltas": {
    "drop_edge": {"test_auc": -0.03, "test_acc": -0.02},
    "bold_noise": {"test_auc": -0.02, "test_acc": -0.01}
  }
}
```

**CLI:**

```bash
python -m brain_gcn.ablation_cli \
  --ablation_components drop_edge bold_noise \
  --ablation_output_dir results/ablations \
  --model_name graph_temporal
```

---

## 4. Extended Evaluation Metrics

### Module: `brain_gcn/utils/evaluation.py`

Comprehensive classification metrics beyond basic accuracy/AUC.

**Metrics provided:**

| Metric | Description | Use Case |
|--------|-------------|----------|
| Sensitivity | ASD recall — can we find ASD patients? | Screening |
| Specificity | TD recall — can we correctly identify controls? | Specificity |
| Precision | ASD precision — false positive rate | Treatment planning |
| F1 Score | Harmonic mean of precision/recall | Balanced evaluation |
| MCC | Matthews Correlation Coefficient | Imbalanced data |
| Kappa | Cohen's Kappa — inter-rater agreement | Clinical validation |

**API:**

```python
from brain_gcn.utils.evaluation import (
    compute_metrics,
    compute_roc_curve,
    compute_pr_curve,
    compute_confusion_matrix,
    StatisticalTester,
)

# Compute all metrics
metrics = compute_metrics(probs, labels)
print(f"AUC: {metrics.auc:.4f}")
print(f"Sensitivity: {metrics.sensitivity:.4f}")
print(f"Specificity: {metrics.specificity:.4f}")

# ROC curve
roc = compute_roc_curve(probs, labels)
# Returns: {fpr, tpr, thresholds, auc}

# Precision-Recall curve
pr = compute_pr_curve(probs, labels)
# Returns: {precision, recall, thresholds, ap}

# Confusion matrix
cm = compute_confusion_matrix(probs, labels)
print(f"TP={cm.true_positives}, FP={cm.false_positives}")
print(f"FN={cm.false_negatives}, TN={cm.true_negatives}")

# Bootstrap confidence intervals
ci_lower, ci_est, ci_upper = StatisticalTester.bootstrap_ci(
    metric_fn=lambda p, l: compute_metrics(p, l).auc,
    probs=probs,
    labels=labels,
    n_bootstrap=1000,
    ci=0.95,
)
print(f"AUC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**CLI:**

```bash
python -m brain_gcn.eval_cli \
  --eval_checkpoint checkpoints/graph_temporal/brain-gcn-050-0.856.ckpt \
  --eval_output_dir results/evaluation \
  --eval_plot_roc \
  --eval_plot_pr \
  --eval_bootstrap_ci \
  --eval_ci_n_bootstrap 1000
```

**Plots generated:**
- `roc_curve.png` — ROC curve with AUC
- `pr_curve.png` — Precision-Recall curve with AP

---

## 5. Experiment Tracking

### Module: `brain_gcn/utils/tracking.py`

Logs experiment metadata, hyperparameters, and results.

**API:**

```python
from brain_gcn.utils.tracking import ExperimentTracker, ExperimentMetadata, RunLogger

# Create tracker
tracker = ExperimentTracker(output_dir="experiments")

# Log a run
with RunLogger("run_001", args, tracker) as metadata:
    # Train model
    trainer, task, dm = train_from_args(args)
    
    # Update metadata
    metadata.update_metrics({"test_auc": 0.856, "test_acc": 0.802})
    metadata.set_checkpoint_path("checkpoints/...")

# Save summary
tracker.save_summary()
```

**Output structure:**

```
experiments/
├── run_001/
│   └── metadata.json
├── run_002/
│   └── metadata.json
└── summary.json
```

**Metadata content:**

```json
{
  "run_id": "run_001",
  "timestamp": "2025-04-11T14:35:20.123456",
  "model_name": "graph_temporal",
  "split_strategy": "site_holdout",
  "hyperparameters": {
    "hidden_dim": 64,
    "dropout": 0.5,
    "lr": 0.001,
    "batch_size": 16
  },
  "test_metrics": {
    "test_auc": 0.856,
    "test_acc": 0.802
  },
  "device": "cuda",
  "pytorch_version": "2.0.0"
}
```

---

## Typical Workflows

### Workflow 1: Find Best Hyperparameters

```bash
# 1. Run HPO
python -m brain_gcn.hpo_cli \
  --hpo_n_trials 20 \
  --hpo_objective test_auc \
  --max_epochs 50 \
  --split_strategy site_holdout

# 2. Review best params (in results/hpo_summary.json)
# 3. Train final model with best params
python -m brain_gcn.main \
  --model_name graph_temporal \
  --hidden_dim 128 \
  --dropout 0.3 \
  --lr 0.0005 \
  --max_epochs 200
```

### Workflow 2: Validate with Cross-Validation

```bash
# Run 5-fold CV to estimate generalization
python -m brain_gcn.cv_cli \
  --cv_n_splits 5 \
  --model_name graph_temporal \
  --max_epochs 50

# View aggregated metrics in results/cv/cv_summary.json
```

### Workflow 3: Understand Component Importance

```bash
# Run ablation to see which components matter
python -m brain_gcn.ablation_cli \
  --ablation_components drop_edge bold_noise graph \
  --model_name graph_temporal \
  --max_epochs 50

# Review impact in results/ablations/ablation_results.json
```

### Workflow 4: Comprehensive Model Evaluation

```bash
# 1. Train model
python -m brain_gcn.main \
  --model_name graph_temporal \
  --max_epochs 200 \
  --test

# 2. Evaluate with extended metrics
python -m brain_gcn.eval_cli \
  --eval_checkpoint checkpoints/graph_temporal/brain-gcn-200-*.ckpt \
  --eval_plot_roc \
  --eval_plot_pr \
  --eval_bootstrap_ci

# 3. Review results in results/evaluation/
```

---

## Dependencies

New dependencies added:
- `optuna>=3.0.0` — hyperparameter optimization
- `scikit-learn>=1.0.0` — cross-validation, metrics
- `matplotlib>=3.5.0` — curve plotting
- `scipy>=1.8.0` — statistical tests

Install with:
```bash
pip install optuna scikit-learn matplotlib scipy
```

---

## Summary of Improvements

| Feature | Scope | Impact |
|---------|-------|--------|
| HPO | Automated hyperparameter search | Reduces manual tuning, finds better configs |
| CV | K-fold + LOSO validation | Better generalization estimates |
| Ablation | Component importance analysis | Understand what drives performance |
| Metrics | Sensitivity, specificity, MCC, Kappa | Clinical relevance |
| Tracking | Experiment metadata logging | Reproducibility, comparison |

---

## Next Steps (Optional Enhancements)

1. **Ensemble Analysis**: Compare predictions across trained models
2. **Visualization Dashboard**: Interactive results browser
3. **Model Interpretability**: Feature importance, attention visualization
4. **Hyperband**: Accelerated HPO with early stopping
5. **Multi-objective Optimization**: Pareto efficiency (AUC vs latency)

