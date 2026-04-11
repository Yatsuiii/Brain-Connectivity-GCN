"""
Download ABIDE I preprocessed ROI time series directly from AWS S3.

Bypasses nilearn entirely — uses boto3 to download files from the public
FCP-INDI S3 bucket with parallel threads and automatic resume.

S3 layout:
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/
      <SITE>_<SUBID>_rois_cc200.1D   (one per subject, ~500 KB each)
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from sklearn.utils import Bunch

log = logging.getLogger(__name__)

# S3 coordinates (public bucket — no credentials needed)
S3_BUCKET       = "fcp-indi"
S3_TS_PREFIX    = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/"
S3_PHENO_KEY    = "data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"

SUBJECT_ID_COL  = "SUB_ID"
LABEL_COL       = "DX_GROUP"   # 1 = ASD, 2 = Typical Control
SITE_COL        = "SITE_ID"

_DEFAULT_WORKERS = 8


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _download_one(key: str, dest: Path) -> bool:
    """Download a single S3 object to dest. Returns True on success."""
    if dest.exists():
        return True  # already cached
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        _s3_client().download_file(S3_BUCKET, key, str(tmp))
        tmp.rename(dest)
        return True
    except Exception as exc:
        log.debug("Failed to download %s: %s", key, exc)
        tmp.unlink(missing_ok=True)
        return False


def fetch_abide(
    data_dir: str | Path,
    n_subjects: int | None = None,
    n_workers: int = _DEFAULT_WORKERS,
    **_kwargs,                     # absorb legacy nilearn kwargs silently
) -> Bunch:
    """
    Download ABIDE I cc200 time series from S3 and return a Bunch.

    Parameters
    ----------
    data_dir   : root cache directory (files stored under data_dir/abide_s3/)
    n_subjects : max subjects to download (None = all ~1102)
    n_workers  : parallel download threads

    Returns
    -------
    Bunch with .rois_cc200 (list of arrays) and .phenotypic (DataFrame)
    """
    data_dir = Path(data_dir)
    ts_dir   = data_dir / "abide_s3" / "rois_cc200"
    ts_dir.mkdir(parents=True, exist_ok=True)

    s3 = _s3_client()

    # --- 1. Phenotypic CSV --------------------------------------------------
    pheno_path = data_dir / "abide_s3" / "phenotypic.csv"
    if not pheno_path.exists():
        log.info("Downloading phenotypic CSV from S3 ...")
        s3.download_file(S3_BUCKET, S3_PHENO_KEY, str(pheno_path))
    pheno = pd.read_csv(pheno_path)
    log.info("Phenotypic CSV: %d subjects.", len(pheno))

    # --- 2. List available .1D keys -----------------------------------------
    log.info("Listing S3 objects ...")
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_TS_PREFIX):
        all_keys += [
            o["Key"] for o in page.get("Contents", [])
            if o["Key"].endswith("_rois_cc200.1D")
        ]
    log.info("S3 bucket: %d subjects available.", len(all_keys))

    if n_subjects:
        all_keys = all_keys[:n_subjects]

    # --- 3. Parallel download -----------------------------------------------
    n_already = sum(1 for k in all_keys if (ts_dir / Path(k).name).exists())
    n_needed  = len(all_keys) - n_already
    log.info("Downloading %d subjects (%d already cached) with %d threads ...",
             n_needed, n_already, n_workers)

    def _dl(key):
        return _download_one(key, ts_dir / Path(key).name)

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_dl, k): k for k in all_keys}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            if not fut.result():
                failed += 1
            if done % 50 == 0 or done == len(all_keys):
                log.info("  %d / %d downloaded (%d failed)", done, len(all_keys), failed)

    if failed:
        log.warning("%d subjects failed to download and will be skipped.", failed)

    # --- 4. Build subject id → file map from phenotypic CSV -----------------
    # Filename: <SITE>_<SUB_ID>_rois_cc200.1D  (SUB_ID zero-padded to 7 digits)
    sub_id_to_file: dict[str, Path] = {}
    for f in ts_dir.glob("*_rois_cc200.1D"):
        stem  = f.stem.replace("_rois_cc200", "")   # e.g. "PITT_0050003"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            sub_id_to_file[parts[1]] = f             # "0050003" → path

    # --- 5. Pair arrays with phenotypic rows --------------------------------
    arrays, rows = [], []
    for _, row in pheno.iterrows():
        sub_id = str(int(row[SUBJECT_ID_COL])).zfill(7)
        if sub_id not in sub_id_to_file:
            continue
        try:
            bold = np.loadtxt(sub_id_to_file[sub_id], dtype=np.float32)
            arrays.append(bold)
            rows.append(row)
        except Exception as exc:
            log.debug("Could not load %s: %s", sub_id_to_file[sub_id], exc)

    pheno_out = pd.DataFrame(rows).reset_index(drop=True)
    log.info("Built dataset: %d subjects paired with phenotypic data.", len(arrays))
    return Bunch(rois_cc200=arrays, phenotypic=pheno_out)


def get_label(phenotypic_row) -> int:
    """DX_GROUP: 1 = ASD, 2 = Typical Control  →  ASD=1, TC=0"""
    dx = int(phenotypic_row[LABEL_COL])
    assert dx in (1, 2), f"Unexpected DX_GROUP value: {dx}. Must be 1 (ASD) or 2 (TC)."
    return 1 if dx == 1 else 0


def extract_subjects(
    dataset: Bunch,
    min_timepoints: int = 100,
) -> list[dict]:
    """
    Validate and pair each subject's BOLD array with its label and metadata.

    Returns list of dicts with keys:
        subject_id, site, label, bold (np.ndarray T×N)
    """
    pheno  = dataset.phenotypic
    arrays = dataset.rois_cc200

    subjects, dropped = [], 0

    for i, bold in enumerate(arrays):
        bold = np.array(bold, dtype=np.float32)

        if bold.ndim != 2:
            log.warning("Subject %d: unexpected shape %s — skipping.", i, bold.shape)
            dropped += 1
            continue

        if not np.isfinite(bold).all():
            log.warning("Subject %d: NaN/Inf values — skipping.", i)
            dropped += 1
            continue

        if bold.shape[0] < min_timepoints:
            log.debug("Subject %d: only %d TRs (min=%d) — skipping.",
                      i, bold.shape[0], min_timepoints)
            dropped += 1
            continue

        row = pheno.iloc[i]
        subjects.append({
            "subject_id":   str(row[SUBJECT_ID_COL]),
            "site":         str(row[SITE_COL]),
            "label":        get_label(row),
            "bold":         bold,
            "n_timepoints": bold.shape[0],
        })

    log.info("Kept %d subjects, dropped %d.", len(subjects), dropped)
    return subjects
