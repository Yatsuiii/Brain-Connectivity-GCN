---
language: en
license: apache-2.0
tags:
  - neuroscience
  - ASD
  - fMRI
  - graph-neural-network
  - brain-connectivity
  - amd-mi300x
  - rocm
  - pytorch
pipeline_tag: graph-ml
datasets:
  - ABIDE-I
---

# brain-connect-gcn

**Adversarial Brain Mode Network — site-invariant ASD classification from resting-state fMRI.**

20-model LOSO ensemble trained on ABIDE I (1,102 subjects, 20 acquisition sites, 3 atlases). Achieves **AUC 0.7298** on CC200 cross-site evaluation — every subject is test-only exactly once, trained on all other sites.

Live demo: [BrainConnect-ASD Space](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/BrainConnect-ASD)  
Checkpoints: stored in the Space repo under `checkpoints/`

---

## Architecture

```
Input: BOLD time-series (196 TRs × N ROIs)
  ↓
Preprocessing: z-score → FC matrix (Pearson r → Fisher z) → adjacency thresholding
  ↓
Brain Mode Decomposition: K=32 learnable spectral modes (graph wavelet basis)
  ↓
GCN Encoder: 2-layer graph convolution, hidden_dim=128, dropout=0.3
  ↓
Gradient Reversal Layer (Ganin 2016): removes site-specific scanner artifacts
  ↓
ASD Classifier head → p(ASD) ∈ [0, 1]
  ↓
20-model ensemble mean → final prediction
```

| Component | Detail |
|---|---|
| **Total parameters** | ~105K |
| **Learnable modes** | K = 32 |
| **Hidden dim** | 128 |
| **Dropout** | 0.3 |
| **Atlases** | CC200 (200 ROIs) · AAL (116 ROIs) · Harvard-Oxford (111 ROIs) |
| **Loss** | Cross-entropy + GRL adversarial site loss |
| **Training** | 150 epochs · batch 32 · lr 5e-4 · AdamW |

---

## Why Adversarial Training?

ABIDE I was collected across 20 sites with different scanners, protocols, and demographics. A naive classifier learns site identity as a shortcut — it generalizes poorly to unseen institutions.

The **Gradient Reversal Layer** (GRL) flips the gradient sign for the site prediction head during backprop, forcing the encoder to produce representations that are *maximally confusing* to a site classifier. Scanner artifacts and acquisition-specific signals cannot leak into the ASD prediction.

This is validated empirically: the LOSO protocol uses a model trained on 19 sites to predict the 20th — it has never seen that institution's data, scanner, or demographic distribution.

---

## Results

| Metric | Value |
|---|---|
| **CC200 LOSO AUC** | **0.7298** |
| HO LOSO AUC | 0.7212 |
| AAL LOSO AUC | 0.6959 |
| Subjects | 1,102 (ABIDE I) |
| Sites | 20 |
| Evaluation | Cross-site LOSO — every subject is test-only |

### vs. Published ABIDE Baselines

| Model | AUC | Split |
|---|---|---|
| SVM + FC (Plitt 2015) | 0.71 | Same-site |
| BrainNetCNN (Kawahara 2017) | 0.74 | Same-site |
| GCN + FC (Ktena 2018) | 0.70 | Same-site |
| ABIDE site-specific SVM | 0.76 | Same-site |
| **BrainConnect-ASD (ours)** | **0.7298** | **Cross-site LOSO** |

All prior baselines use same-site train/test splits — a fundamentally easier evaluation. Cross-site LOSO is the clinically relevant bar.

---

## Training Hardware

Trained on **AMD MI300X** (192GB HBM3) via DigitalOcean, ROCm 7.0, PyTorch 2.5.1+rocm6.2.

All 20 LOSO folds across 3 atlases (60 total training runs) were executed in parallel on the MI300X — a workload that would take days on CPU completed in hours.

End-to-end inference (preprocessing + 20-model ensemble): **~20ms per subject** on MI300X.

---

## Usage

```python
# Checkpoints are in the BrainConnect-ASD Space repo
from huggingface_hub import hf_hub_download
from brain_gcn.tasks import ClassificationTask

ckpt = hf_hub_download(
    repo_id="lablab-ai-amd-developer-hackathon/BrainConnect-ASD",
    filename="checkpoints/cc200/adv_brain_mode_k32_site_cc200_loso_nyu/best.ckpt",
    repo_type="space"
)
task = ClassificationTask.load_from_checkpoint(ckpt, map_location="cpu", strict=False)
task.model.eval()

# bw_t: (1, 30, 200) windowed BOLD · adj_t: (1, 200, 200) FC adjacency
import torch
with torch.no_grad():
    out = task.model(bw_t, adj_t)
    p_asd = torch.softmax(out, -1)[0, 1].item()
```

---

## Citation

```
BrainConnect-ASD — AMD Developer Hackathon 2026
Raghav Aryen · lablab.ai · AMD MI300X
https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/BrainConnect-ASD
```
