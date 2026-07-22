# Historical implementation notes

This file summarizes an early development phase of BrainConnect-ASD. It is
retained for provenance and should not be used as a source of benchmark claims.

The phase introduced:

- synthetic ABIDE-shaped data for pipeline smoke tests;
- stratified and site-aware split utilities;
- class weighting and train-only population adjacency construction;
- graph and temporal model variants;
- Optuna hyperparameter-search support;
- early stopping, checkpoint ensembling, and evaluation metrics; and
- experiment, ablation, and cross-validation helpers.

Synthetic-data accuracy is useful only for verifying that the software runs. It
does not estimate performance on ABIDE or on clinical populations. Increasing
model capacity, the number of subjects, or hyperparameter trials does not imply
a particular target accuracy.

For current results and limitations, use [`README.md`](README.md) and
[`MODEL_CARD.md`](MODEL_CARD.md). Checked-in CSV files under `results/` are a mix
of historical smoke tests and experiments; each should be interpreted only with
its corresponding configuration and split definition.
