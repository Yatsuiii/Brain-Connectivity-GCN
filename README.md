# BrainConnect-ASD

[![Tests](https://github.com/Yatsuiii/Brain-Connectivity-GCN/actions/workflows/tests.yml/badge.svg)](https://github.com/Yatsuiii/Brain-Connectivity-GCN/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/Hugging%20Face-demo-orange)](https://huggingface.co/spaces/Yatsuiii/BrainConnect-ASD)

BrainConnect-ASD is a research prototype for studying whether adversarial graph
neural networks can reduce acquisition-site confounding in autism spectrum
disorder (ASD) classification from resting-state fMRI. Its headline four-site
LOSO evaluation achieves **0.7872 ROC AUC across 529 held-out subjects**.

The project was developed for the AMD Developer Hackathon 2026 and trained on
ABIDE I. It is intended for reproducible machine-learning research, **not for
clinical diagnosis, screening, or treatment decisions**.

## Research question

Multi-site neuroimaging models can learn scanner and acquisition artifacts
instead of biological signal. BrainConnect-ASD combines a lightweight graph
encoder with a gradient-reversal site classifier to encourage representations
that are less informative about the acquisition site.

```text
BOLD time series (T x 200 ROIs)
        |
        +-- sliding-window features
        +-- Fisher-z functional-connectivity graph
        |
        v
Adversarial Brain Mode Network
        |
        +-- ASD/typical-control research classifier
        +-- site-adversarial head (training only)
```

Key implementation details:

- CC200 functional atlas with 200 regions of interest
- 32 learnable graph modes
- approximately 105,000 trainable parameters
- gradient reversal for site-adversarial training
- leave-one-site-out (LOSO) evaluation

## Results

The headline evaluation covers NYU, USM, UCLA, and UM. Its aggregate performance
is shown first; the broader 20-site experiment is included as a separate
generalization benchmark and powers the current multi-site demo.

| Evaluation | Subjects | ROC AUC |
|---|---:|---:|
| **Four-site LOSO evaluation (NYU, USM, UCLA, UM)** | **529** | **0.7872** |
| Broader ABIDE I 20-site LOSO experiment | 1,102 | 0.7298 |

The two rows describe different evaluation scopes and should not be pooled or
treated as interchangeable. Each LOSO fold trains without the evaluation site.
ABIDE II external evaluation is planned but has not yet been reported.

These results do not establish clinical validity. Performance can be affected
by demographic imbalance, motion, preprocessing choices, site composition, and
dataset shift.

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/Yatsuiii/Brain-Connectivity-GCN.git
cd Brain-Connectivity-GCN
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

PyTorch installation can be platform-specific. If the default wheel is not
suitable for your CPU, CUDA, or ROCm environment, install PyTorch using its
platform instructions before installing this package.

## Usage

Run inference on a preprocessed CC200 ROI time-series file:

```bash
python predict.py subject_rois_cc200.1D --verbose
```

Evaluate the published ensemble on ABIDE II after reviewing the download and
dataset-use requirements:

```bash
python eval_abide2.py --n-subjects 50
python eval_abide2.py
```

Training and experiment entry points are documented in
[MODELS_AND_PIPELINE.md](MODELS_AND_PIPELINE.md) and
[EXPERIMENTS.md](EXPERIMENTS.md). The detailed research-use model card is in
[MODEL_CARD.md](MODEL_CARD.md).

## Repository layout

```text
brain_gcn/             model, training, and evaluation modules
tests/                 unit and integration tests
hf_space/              Hugging Face demo source
results/               checked-in experiment summaries
predict.py             single-subject research inference
eval_abide2.py         external-dataset evaluation script
generate_finetune_data.py  synthetic explanation-data generator
```

## Reproducibility and data

ABIDE data are not redistributed in this repository. Users are responsible for
complying with the dataset's access and use terms. Random seeds, preprocessing
choices, split definitions, checkpoint provenance, and exclusions should be
reported alongside any derived result.

The generated LLM examples in `finetune_data/` are synthetic templates. They
are not clinician-authored reports, patient records, or evidence of clinical
validation. They must not be used to infer subject-specific neurobiological
findings without an independently validated attribution method.

## Limitations

- ASD is heterogeneous and cannot be diagnosed from resting-state fMRI alone.
- ABIDE is a retrospective research dataset, not a prospective clinical cohort.
- Site-adversarial training does not prove removal of every scanner, motion, or
  demographic confound.
- The headline result has not yet been replicated on ABIDE II in this repo.
- Saliency and generated text are explanatory research aids, not causal or
  clinical interpretations.

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
and include tests and reproducibility details with behavioral changes.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Until a paper is
available, cite the software repository and the exact release or commit used.

## License

Licensed under the [Apache License 2.0](LICENSE). Dataset and third-party model
licenses remain separate.
