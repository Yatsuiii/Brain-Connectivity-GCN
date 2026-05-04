#!/usr/bin/env python3
"""
Predict ASD vs Typical Control from a raw ABIDE CC200 .1D time-series file.

Ensemble of 4 adversarial brain-mode GCNs trained with leave-one-site-out
cross-validation (LOSO). Each model held out a different scanner site —
the ensemble is site-agnostic by design.

LOSO evaluation across 529 unseen subjects (4 institution clusters):
  NYU  (n=184)  AUC 0.7924
  USM  (n=101)  AUC 0.7855
  UCLA (n= 99)  AUC 0.8086
  UM   (n=145)  AUC 0.7624
  Mean          AUC 0.7872 ± 0.019

Usage:
    python predict.py subject.1D                   # raw ABIDE .1D file
    python predict.py data/processed/50002.npz     # preprocessed .npz
    python predict.py data/processed/              # batch over all .npz files
    python predict.py data/abide_s3/rois_cc200/    # batch over all .1D files
    python predict.py subject.1D --verbose         # per-model breakdown
    python predict.py --ui                         # launch Gradio web demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── preprocessing constants (must match training) ──────────────────────────
_WINDOW_LEN   = 50
_STEP         = 3
_MAX_WINDOWS  = 30
_FC_THRESHOLD = 0.2

# ── LOSO ensemble checkpoints ──────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
_CHECKPOINTS = [
    (
        _REPO / "checkpoints/adv_brain_mode_k16_site_site_nyu"
               / "brain-gcn-epoch=008-val_auc=0.776.ckpt",
        "NYU",
    ),
    (
        _REPO / "checkpoints/adv_brain_mode_k16_site_loso_usm"
               / "brain-gcn-epoch=004-val_auc=0.780.ckpt",
        "USM",
    ),
    (
        _REPO / "checkpoints/adv_brain_mode_k16_site_loso_ucla_both"
               / "brain-gcn-epoch=005-val_auc=0.738.ckpt",
        "UCLA",
    ),
    (
        _REPO / "checkpoints/adv_brain_mode_k16_site_loso_um_both"
               / "brain-gcn-epoch=060-val_auc=0.851.ckpt",
        "UM",
    ),
]


# ── preprocessing ──────────────────────────────────────────────────────────

def _zscore(bold: np.ndarray) -> np.ndarray:
    mean = bold.mean(axis=0, keepdims=True)
    std  = bold.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((bold - mean) / std).astype(np.float32)


def _compute_fc(bold: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(bold.T).astype(np.float32)
    np.nan_to_num(fc, copy=False)
    return fc


def _bold_windows(bold: np.ndarray) -> np.ndarray:
    T, N = bold.shape
    starts = list(range(0, T - _WINDOW_LEN + 1, _STEP))
    wins = np.stack([bold[s:s + _WINDOW_LEN].std(axis=0) for s in starts]).astype(np.float32)
    if len(wins) >= _MAX_WINDOWS:
        return wins[:_MAX_WINDOWS]
    pad = np.repeat(wins[-1:], _MAX_WINDOWS - len(wins), axis=0)
    return np.concatenate([wins, pad])


def _preprocess(bold: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    bold = _zscore(bold)
    fc   = _compute_fc(bold)
    fc   = np.arctanh(np.clip(fc, -0.9999, 0.9999))       # Fisher Z
    mask = np.abs(fc) >= _FC_THRESHOLD
    adj  = np.where(mask, fc, 0.0).astype(np.float32)     # threshold, preserve sign
    bw   = _bold_windows(bold)
    return torch.FloatTensor(bw).unsqueeze(0), torch.FloatTensor(adj).unsqueeze(0)


def _load_1d(path: Path) -> np.ndarray:
    bold = np.loadtxt(path, dtype=np.float32)
    if bold.ndim != 2:
        raise ValueError(f"Expected 2D (T×N), got shape {bold.shape}")
    if bold.shape[1] != 200:
        raise ValueError(f"Expected 200 ROIs (CC200), got {bold.shape[1]} columns")
    if bold.shape[0] < _WINDOW_LEN + _STEP:
        raise ValueError(f"Only {bold.shape[0]} TRs (need ≥ {_WINDOW_LEN + _STEP})")
    return bold


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load already-preprocessed .npz — returns (bold_windows, mean_fc) before Fisher Z."""
    d = np.load(path, allow_pickle=True)
    bw = d["bold_windows"].astype(np.float32)   # (W, N) already computed
    fc = d["mean_fc"].astype(np.float32)         # (N, N) raw Pearson
    return bw, fc


# ── model loading ──────────────────────────────────────────────────────────

def _load_ensemble(device: str, verbose: bool = False) -> list[tuple]:
    from brain_gcn.tasks import ClassificationTask

    models = []
    for ckpt_path, site in _CHECKPOINTS:
        if not ckpt_path.exists():
            print(f"  [warn] missing checkpoint: {ckpt_path.name}", file=sys.stderr)
            continue
        task = ClassificationTask.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False
        )
        task.eval().to(device)
        models.append((task, site))
        if verbose:
            print(f"  ✓ [{site}] {ckpt_path.name}")

    if not models:
        raise RuntimeError("No checkpoints found. Run from the repo root.")
    return models


# ── inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_file(
    path: Path,
    models: list[tuple],
    device: str,
    verbose: bool = False,
) -> dict:
    if path.suffix == ".npz":
        bw_np, fc_np = _load_npz(path)
        fc_np = np.arctanh(np.clip(fc_np, -0.9999, 0.9999))
        mask  = np.abs(fc_np) >= _FC_THRESHOLD
        adj_np = np.where(mask, fc_np, 0.0).astype(np.float32)
        if len(bw_np) >= _MAX_WINDOWS:
            bw_np = bw_np[:_MAX_WINDOWS]
        else:
            pad   = np.repeat(bw_np[-1:], _MAX_WINDOWS - len(bw_np), axis=0)
            bw_np = np.concatenate([bw_np, pad])
        bw  = torch.FloatTensor(bw_np).unsqueeze(0)
        adj = torch.FloatTensor(adj_np).unsqueeze(0)
    else:
        bold = _load_1d(path)
        bw, adj = _preprocess(bold)
    bw  = bw.to(device)
    adj = adj.to(device)

    per_model = []
    for task, site in models:
        logits   = task(bw, adj)
        p_asd    = torch.softmax(logits, dim=-1)[0, 1].item()
        per_model.append((site, p_asd))
        if verbose:
            lbl = "ASD" if p_asd > 0.5 else "TC "
            print(f"    [{site:>4}] {lbl}  p={p_asd:.3f}")

    p_mean     = float(np.mean([p for _, p in per_model]))
    label      = "ASD" if p_mean > 0.5 else "Typical Control"
    confidence = p_mean if p_mean > 0.5 else 1.0 - p_mean

    return {
        "file":       str(path.name),
        "prediction": label,
        "p_asd":      p_mean,
        "confidence": confidence,
        "per_model":  per_model,
    }


# ── Gradio UI ──────────────────────────────────────────────────────────────

def _launch_gradio(models: list[tuple], device: str) -> None:
    try:
        import gradio as gr
    except ImportError:
        print("Install gradio first:  pip install gradio", file=sys.stderr)
        sys.exit(1)

    import tempfile

    def _infer(file_obj) -> str:
        if file_obj is None:
            return "Upload a .1D file to continue."
        path = Path(file_obj.name)
        try:
            result = predict_file(path, models, device, verbose=False)
            lines = [
                f"**{result['prediction']}**  ({result['confidence']*100:.1f}% confidence)",
                f"p(ASD) = {result['p_asd']:.3f}",
                "",
                "Per-model breakdown:",
            ]
            for site, p in result["per_model"]:
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                lbl = "ASD" if p > 0.5 else "TC "
                lines.append(f"  {site:>4}  {lbl}  {bar}  {p:.3f}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"

    demo = gr.Interface(
        fn=_infer,
        inputs=gr.File(label="Upload CC200 .1D file", file_types=[".1D"]),
        outputs=gr.Textbox(label="Prediction", lines=10),
        title="Brain Connectivity ASD Predictor",
        description=(
            "Upload an ABIDE CC200 ROI time series (.1D format, 200 columns).\n"
            "Ensemble of 4 adversarial GCNs trained with leave-one-site-out CV.\n"
            "LOSO mean AUC = 0.7872 across 529 unseen subjects."
        ),
        allow_flagging="never",
    )
    demo.launch()


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASD/TC prediction from ABIDE CC200 .1D time series"
    )
    parser.add_argument("input", nargs="?", type=Path,
                        help=".1D file or directory of .1D files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-model probabilities")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--ui", action="store_true",
                        help="Launch Gradio web interface instead of CLI")
    args = parser.parse_args()

    if args.ui and args.input:
        parser.error("--ui and input path are mutually exclusive")
    if not args.ui and args.input is None:
        parser.error("Provide an input file or directory, or use --ui")

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    print(f"Device : {device}")
    print(f"Loading ensemble ({len(_CHECKPOINTS)} LOSO models)...")
    models = _load_ensemble(device, verbose=args.verbose)
    print(f"  → {len(models)} model(s) ready\n")

    if args.ui:
        _launch_gradio(models, device)
        return

    # Collect files
    target = args.input
    if target.is_dir():
        files = sorted(target.glob("*_rois_cc200.1D"))
        if not files:
            files = sorted(target.glob("*.1D"))
        if not files:
            files = sorted(target.glob("*.npz"))
        if not files:
            print(f"No .1D or .npz files found in {target}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(files)} subjects in {target}\n")
    else:
        files = [target]

    errors = 0
    for fp in files:
        if args.verbose and len(files) > 1:
            print(f"── {fp.name}")
        try:
            r = predict_file(fp, models, device, verbose=args.verbose)
            pct = r["confidence"] * 100
            if len(files) == 1:
                print(f"Prediction : {r['prediction']}")
                print(f"Confidence : {pct:.1f}%  (p_ASD = {r['p_asd']:.3f})")
                if not args.verbose:
                    print("\nRun with --verbose to see per-model breakdown.")
            else:
                tag = "ASD" if r["p_asd"] > 0.5 else "TC "
                print(f"{fp.name:<50}  {tag}  {pct:5.1f}%  (p={r['p_asd']:.3f})")
        except Exception as exc:
            print(f"ERROR  {fp.name}: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
