---
title: BrainConnect ASD
emoji: ⚡
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: "3.13"
app_file: app.py
pinned: false
license: apache-2.0
short_description: ASD detection from brain connectivity
tags:
  - amd
  - amd-hackathon-2026
  - medical-ai
  - brain-connectivity
  - gradio
---

# BrainConnect-ASD

Site-invariant Autism Spectrum Disorder classification from resting-state fMRI, built for the AMD Developer Hackathon 2026.

## Models

| Model | Description |
|---|---|
| [Yatsuiii/brain-connect-gcn](https://huggingface.co/Yatsuiii/brain-connect-gcn) | Adversarial Brain Mode Network — 20-model LOSO GCN ensemble, AUC 0.7298 cross-site |
| [Yatsuiii/asd-interpreter-merged](https://huggingface.co/Yatsuiii/asd-interpreter-merged) | Qwen2.5-7B fine-tuned clinical interpreter — generates natural language reports from saliency scores |

## Pipeline

```
fMRI (.1D) → preprocessing → FC matrix → GCN ensemble (20 models) → p(ASD)
                                                    ↓
                                         gradient saliency → Qwen2.5-7B → clinical report
```

- **60 total models** trained across 3 atlases (CC200, AAL, Harvard-Oxford) × 20 LOSO folds
- **AMD MI300X** (ROCm 7.0) used for all training and LLM inference
- **~20ms** end-to-end inference per subject (preprocessing + 20-model ensemble)
- **1,102 subjects** · 20 acquisition sites · cross-site evaluation only
