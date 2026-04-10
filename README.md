# Brain-Connectivity-GCN

Dynamic functional connectivity models for ASD vs TD classification from
resting-state fMRI.

This repo is now centered on the `brain_gcn` package. The old external
graph-temporal codebase has been moved under `references/graph_temporal_reference/`
as background material only; the active implementation does not need to follow it
strictly.

## Current Path

- Preprocess ABIDE/CC200 BOLD time series into per-subject `.npz` files.
- Build static or dynamic functional-connectivity adjacency matrices.
- Train a subject-level graph-temporal classifier with ROI readout.
- Compare stronger brain-specific baselines and ablations as the project grows.

## Smoke Test

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache venv/bin/python -m brain_gcn.main \
  --max_epochs 1 \
  --max_windows 5 \
  --hidden_dim 8 \
  --dropout 0.1 \
  --batch_size 4 \
  --num_workers 0 \
  --accelerator cpu \
  --devices 1
```

## Model Modes

- `graph_temporal`: graph projection per window + GRU temporal encoder.
- `gcn`: graph-only baseline over each ROI's average window signal.
- `gru`: temporal baseline over ROI vectors, with no graph message passing.
- `fc_mlp`: static connectivity MLP over the upper triangle of each subject's FC
  adjacency. Use it with `--no-use_population_adj`.

To use one FC graph per sliding window with `graph_temporal`, run with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache venv/bin/python -m brain_gcn.main \
  --model_name graph_temporal \
  --no-use_population_adj \
  --use_dynamic_adj_sequence
```

## Experiment Comparison

Run all baseline modes on the same split and write a CSV summary:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache venv/bin/python -m brain_gcn.experiments \
  --models fc_mlp gru gcn graph_temporal \
  --dynamic_graph_temporal \
  --max_epochs 50 \
  --batch_size 16 \
  --results_csv results/experiment_summary.csv
```

For a stricter held-out-site evaluation, add:

```bash
--split_strategy site_holdout
```

With the current 100-subject cache, the automatic site split trains on PITT,
validates on OHSU, and tests on OLIN. You can override it with `--val_site` and
`--test_site`.
