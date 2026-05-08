#!/usr/bin/env python3
"""
Zero-shot evaluation of the ABIDE-I trained ensemble on ABIDE II.

Downloads ABIDE II CC200 ROI time series from the Preprocessed Connectomes
Project S3 bucket (no NIfTI, no FSL/ANTs required), runs the 4-model ensemble,
and reports AUC + accuracy stratified by site.

Usage:
    python eval_abide2.py                        # download + eval (default)
    python eval_abide2.py --data-dir /tmp/abide2 # custom cache dir
    python eval_abide2.py --skip-download        # re-use already-downloaded files
    python eval_abide2.py --n-subjects 100       # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
import os
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

# ── reuse preprocessing from predict.py ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import _load_ensemble, predict_file, _CHECKPOINTS   # noqa: E402

# ── ABIDE II S3 constants ──────────────────────────────────────────────────
_PHENO_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/"
    "RawData/Phenotypic_V1_0b.csv"
)
_TS_BASE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/Outputs/"
    "cpac/filt_global/rois_cc200/{subject_id}_rois_cc200.1D"
)
# Some ABIDE II subjects are named with site prefix in a sub-folder structure;
# we try the flat naming first, then the site-prefixed one.
_TS_BASE_ALT = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/Outputs/"
    "cpac/filt_global/rois_cc200/{site_id}_{subject_id}_rois_cc200.1D"
)

_DX_ASD = 1   # DSM label in phenotypic CSV
_DX_TC  = 2


# ── phenotypic download ────────────────────────────────────────────────────

def _download_phenotypic(data_dir: Path) -> Path:
    pheno_path = data_dir / "Phenotypic_V1_0b.csv"
    if pheno_path.exists():
        print(f"  [cache] phenotypic CSV already at {pheno_path}")
        return pheno_path
    print("  Downloading ABIDE II phenotypic CSV...", end=" ", flush=True)
    urllib.request.urlretrieve(_PHENO_URL, pheno_path)
    print("done.")
    return pheno_path


def _parse_phenotypic(pheno_path: Path, n_subjects: int | None) -> list[dict]:
    import csv
    subjects = []
    with open(pheno_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx = int(row.get("DX_GROUP", 0))
            if dx not in (_DX_ASD, _DX_TC):
                continue
            sid = row.get("SUB_ID", "").strip()
            site = row.get("SITE_ID", "").strip()
            if not sid or not site:
                continue
            subjects.append({
                "sub_id": sid,
                "site":   site,
                "label":  1 if dx == _DX_ASD else 0,   # 1=ASD, 0=TC
            })
    if n_subjects:
        subjects = subjects[:n_subjects]
    return subjects


# ── time series download ───────────────────────────────────────────────────

def _ts_path(data_dir: Path, sub_id: str) -> Path:
    return data_dir / "rois_cc200" / f"{sub_id}_rois_cc200.1D"


def _download_one(sub: dict, data_dir: Path) -> tuple[dict, bool]:
    dst = _ts_path(data_dir, sub["sub_id"])
    if dst.exists():
        return sub, True

    urls = [
        _TS_BASE.format(subject_id=sub["sub_id"]),
        _TS_BASE_ALT.format(site_id=sub["site"], subject_id=sub["sub_id"]),
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, dst)
            return sub, True
        except Exception:
            continue
    return sub, False


def _download_timeseries(subjects: list[dict], data_dir: Path, workers: int = 8) -> list[dict]:
    ts_dir = data_dir / "rois_cc200"
    ts_dir.mkdir(parents=True, exist_ok=True)

    found, missing = [], []
    print(f"  Downloading time series for {len(subjects)} subjects "
          f"({workers} parallel workers)...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_download_one, s, data_dir): s for s in subjects}
        for i, fut in enumerate(as_completed(futures), 1):
            sub, ok = fut.result()
            if ok:
                found.append(sub)
            else:
                missing.append(sub["sub_id"])
            if i % 50 == 0 or i == len(subjects):
                print(f"    {i}/{len(subjects)} processed, {len(found)} found, "
                      f"{len(missing)} missing", flush=True)

    if missing:
        print(f"  [warn] {len(missing)} subjects not found on S3 (may be QC-excluded):")
        print("  " + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))
    return found


# ── evaluation ────────────────────────────────────────────────────────────

def _evaluate(subjects: list[dict], data_dir: Path,
               models: list, device: str) -> dict:
    from sklearn.metrics import roc_auc_score

    results = []
    errors  = 0
    for i, sub in enumerate(subjects, 1):
        fp = _ts_path(data_dir, sub["sub_id"])
        try:
            r = predict_file(fp, models, device)
            results.append({
                **sub,
                "p_asd": r["p_asd"],
                "pred":  1 if r["p_asd"] > 0.5 else 0,
            })
        except Exception as exc:
            errors += 1
        if i % 50 == 0 or i == len(subjects):
            print(f"    {i}/{len(subjects)} predicted ({errors} errors so far)",
                  flush=True)

    if not results:
        print("No valid predictions.", file=sys.stderr)
        sys.exit(1)

    labels = [r["label"] for r in results]
    scores = [r["p_asd"]  for r in results]
    preds  = [r["pred"]   for r in results]

    overall_auc = roc_auc_score(labels, scores)
    overall_acc = np.mean([l == p for l, p in zip(labels, preds)])

    # per-site breakdown
    site_stats = {}
    sites = sorted(set(r["site"] for r in results))
    for site in sites:
        s_res = [r for r in results if r["site"] == site]
        s_lbl = [r["label"] for r in s_res]
        s_scr = [r["p_asd"] for r in s_res]
        s_prd = [r["pred"]  for r in s_res]
        if len(set(s_lbl)) < 2:
            site_stats[site] = {"n": len(s_res), "auc": None,
                                 "acc": np.mean([l == p for l, p in zip(s_lbl, s_prd)])}
        else:
            site_stats[site] = {
                "n":   len(s_res),
                "auc": roc_auc_score(s_lbl, s_scr),
                "acc": np.mean([l == p for l, p in zip(s_lbl, s_prd)]),
            }

    return {
        "n_subjects":   len(results),
        "n_errors":     errors,
        "overall_auc":  overall_auc,
        "overall_acc":  overall_acc,
        "site_stats":   site_stats,
        "results":      results,
    }


def _print_report(ev: dict) -> None:
    print("\n" + "═" * 62)
    print("  ABIDE II — Zero-Shot Evaluation Results")
    print("═" * 62)
    print(f"  Subjects evaluated : {ev['n_subjects']}  ({ev['n_errors']} errors)")
    print(f"  Overall AUC        : {ev['overall_auc']:.4f}")
    print(f"  Overall Accuracy   : {ev['overall_acc']*100:.1f}%")
    print()
    print(f"  {'Site':<20}  {'N':>5}  {'AUC':>6}  {'Acc':>6}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*6}  {'-'*6}")
    for site, st in sorted(ev["site_stats"].items()):
        auc_str = f"{st['auc']:.4f}" if st["auc"] is not None else "  n/a "
        print(f"  {site:<20}  {st['n']:>5}  {auc_str:>6}  {st['acc']*100:>5.1f}%")
    print("═" * 62)
    print()
    print("  Interpretation:")
    print("  These models were trained on ABIDE I (20 sites, 1,102 subjects).")
    print("  ABIDE II subjects were never seen during training or validation.")
    print("  AUC > 0.65 on unseen sites/scanners = genuine generalization.")
    print()


def _save_results(ev: dict, out_path: Path) -> None:
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sub_id", "site", "label", "p_asd", "pred"])
        w.writeheader()
        for r in ev["results"]:
            w.writerow({k: r[k] for k in ["sub_id", "site", "label", "p_asd", "pred"]})
    print(f"  Per-subject results saved to {out_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ABIDE II zero-shot evaluation")
    parser.add_argument("--data-dir",      type=Path, default=Path("/tmp/abide2"),
                        help="Directory to cache downloaded data (default: /tmp/abide2)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip S3 download; use already-cached files")
    parser.add_argument("--n-subjects",    type=int, default=None,
                        help="Limit to first N subjects (for quick testing)")
    parser.add_argument("--workers",       type=int, default=8,
                        help="Parallel download threads (default: 8)")
    parser.add_argument("--device",        default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device")
    parser.add_argument("--out",           type=Path, default=Path("abide2_results.csv"),
                        help="CSV output path for per-subject results")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )

    # ── step 1: phenotypic
    print("\n[1/4] Phenotypic data")
    pheno_path = _download_phenotypic(args.data_dir)
    subjects   = _parse_phenotypic(pheno_path, args.n_subjects)
    n_asd = sum(1 for s in subjects if s["label"] == 1)
    n_tc  = len(subjects) - n_asd
    print(f"  {len(subjects)} subjects: {n_asd} ASD, {n_tc} TC  "
          f"across {len(set(s['site'] for s in subjects))} sites")

    # ── step 2: time series
    print("\n[2/4] Time series files")
    if args.skip_download:
        ts_dir = args.data_dir / "rois_cc200"
        existing = {f.stem.split("_rois_cc200")[0] for f in ts_dir.glob("*.1D")}
        subjects = [s for s in subjects if s["sub_id"] in existing]
        print(f"  Skip-download mode: {len(subjects)} cached subjects found")
    else:
        subjects = _download_timeseries(subjects, args.data_dir, workers=args.workers)

    if not subjects:
        print("No subjects available. Check your data-dir or network.", file=sys.stderr)
        sys.exit(1)

    # ── step 3: load models
    print(f"\n[3/4] Loading ensemble ({len(_CHECKPOINTS)} LOSO models) on {device}...")
    models = _load_ensemble(device, verbose=True)

    # ── step 4: evaluate
    print(f"\n[4/4] Running inference on {len(subjects)} subjects...")
    ev = _evaluate(subjects, args.data_dir, models, device)

    _print_report(ev)
    _save_results(ev, args.out)


if __name__ == "__main__":
    try:
        from sklearn.metrics import roc_auc_score  # noqa: F401
    except ImportError:
        print("Install scikit-learn:  pip install scikit-learn", file=sys.stderr)
        sys.exit(1)
    main()
