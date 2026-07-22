---
language: en
license: apache-2.0
tags:
  - neuroscience
  - ASD
  - fMRI
  - graph-neural-network
  - brain-connectivity
  - pytorch
pipeline_tag: graph-ml
datasets:
  - ABIDE-I
---

# BrainConnect-ASD model card

## Model summary

BrainConnect-ASD is an adversarial graph neural network for retrospective
research on ASD versus typical-control classification from resting-state fMRI.
It uses a gradient-reversal head during training to discourage acquisition-site
information in the learned representation. Its headline four-site LOSO
evaluation achieves **0.7872 ROC AUC across 529 held-out ABIDE I subjects**.

This model is a research artifact. It is not a medical device and must not be
used for diagnosis, screening, prognosis, patient selection, or treatment.

## Intended use

Appropriate uses include:

- research on cross-site robustness in functional-connectivity models;
- reproduction or critique of the reported LOSO experiment;
- methodological comparisons under equivalent subject-level splits; and
- educational demonstrations using non-identifiable research data.

Out-of-scope uses include clinical decision-making, individual risk assessment,
unsupervised deployment, and claims about specific brain mechanisms based only
on model predictions or saliency.

## Architecture

| Component | Configuration |
|---|---|
| Input | preprocessed BOLD time series and FC adjacency |
| Parcellation | CC200 for the primary reported result |
| Graph modes | 32 learnable modes |
| Hidden dimension | 128 |
| Dropout | 0.3 |
| Parameters | approximately 105,000 |
| Objective | ASD classification plus site-adversarial loss |
| Optimizer | AdamW |

## Training and evaluation data

The model was developed with ABIDE I, a retrospective multi-institution
neuroimaging research dataset. The repository does not redistribute ABIDE data.
The reported protocol uses leave-one-site-out evaluation: a model is trained on
the other sites and evaluated on the held-out site.

ABIDE II is not included in the reported model-card result. The repository
contains an evaluation script, but external results should not be claimed until
the full run, exclusions, and output artifacts are published.

## Reported performance

| Evaluation | N | ROC AUC |
|---|---:|---:|
| **Four-site LOSO evaluation: NYU, USM, UCLA, UM** | **529** | **0.7872** |
| Broader ABIDE I 20-site LOSO experiment | 1,102 | 0.7298 |

The 0.7872 value describes the four-site evaluation. The 0.7298 value describes
the broader 20-site experiment used by the current demo. They are different
evaluation scopes and are not directly interchangeable. ROC AUC does not select
an operating threshold or establish clinical utility.

## Limitations and risks

- ASD is clinically and biologically heterogeneous.
- Resting-state fMRI alone is insufficient for an ASD diagnosis.
- ABIDE site composition, demographics, head motion, and preprocessing may
  introduce residual confounding.
- Gradient reversal encourages but does not guarantee site invariance.
- Aggregate AUC can conceal poor performance at individual sites or in
  demographic subgroups.
- The current evidence is retrospective; prospective clinical validation has
  not been performed.
- Generated narrative explanations are based on synthetic templates and are
  not validated clinical reports.

Users should report per-site sample sizes and metrics, uncertainty intervals,
calibration, exclusions, preprocessing, and failed runs when publishing derived
work.

## Ethical considerations

Neurodevelopmental labels are sensitive. Use only data for which appropriate
permissions and governance are in place. Avoid stigmatizing interpretations,
claims of biological determinism, and attempts to infer traits outside the
dataset's documented scope. Do not expose identifiable scans, metadata, model
outputs, or generated reports.

## Checkpoints and demo

- [Hugging Face model](https://huggingface.co/Yatsuiii/brain-connect-gcn)
- [Hugging Face demo](https://huggingface.co/spaces/Yatsuiii/BrainConnect-ASD)

Checkpoint users should record the exact filename, revision, preprocessing
configuration, and source commit.

## Citation

See [`CITATION.cff`](CITATION.cff). The repository currently represents software
and preliminary experimental results, not a peer-reviewed clinical study.
