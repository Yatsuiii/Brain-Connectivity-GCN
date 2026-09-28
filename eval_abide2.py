#!/usr/bin/env python3
"""
Zero-shot evaluation of the ABIDE I-trained CC200 ensemble on ABIDE II.

Implements preregistrations/2026-09-28-abide2-external-validation.md. ABIDE II
has no C-PAC ROI release, so inputs are built from its fMRIPrep outputs (see
brain_gcn/external_validation/abide2.py for the S3 layout and processing).

Three steps, run in order:

    python eval_abide2.py discover
        Cohort from listings and phenotype files. Reads no imaging data and
        loads no model, so it is safe before the preregistration is frozen.

    python eval_abide2.py extract [--n-subjects N] [--workers 4]
        Streams each eligible participant's BOLD (~220 MB), writes an ABIDE I
        format CC200 .1D, then deletes the BOLD. Re-running resumes.

    python eval_abide2.py evaluate --prereg-sha <frozen commit>
        Scores the .1D files with the unmodified predict.py ensemble and applies
        the preregistered analysis. Refuses to run until the plan is frozen.

Outputs go under --data-dir (default data/abide2, git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from brain_gcn.external_validation import abide2, analysis  # noqa: E402
from brain_gcn.external_validation.prereg import (  # noqa: E402
    PreregistrationNotFrozen, require_frozen,
)

COHORT_CSV = "cohort.csv"
SITE_TR_JSON = "site_tr.json"
EXTRACTION_CSV = "extraction.csv"
PREDICTIONS_CSV = "predictions.csv"
METRICS_JSON = "metrics.json"
ROI_DIR = "rois_cc200"
WORK_DIR = "work"

COHORT_FIELDS = ["participant_id", "site_id", "dx_group", "age", "sex", "processed",
                 "run_stem", "bold_key", "confounds_key", "bold_bytes", "exclusion"]
EXTRACTION_FIELDS = ["participant_id", "path", "exclusion", "n_volumes", "tr",
                     "site_tr", "n_empty_rois", "mean_fd", "error"]


# ── csv helpers ────────────────────────────────────────────────────────────

def _write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ── discover ───────────────────────────────────────────────────────────────

def cmd_discover(args: argparse.Namespace) -> int:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print("Listing ABIDE II fMRIPrep outputs and site phenotype files...")
    cohort, site_tr = abide2.discover()

    rows = []
    for entry in cohort:
        person, run = entry.participant, entry.run
        rows.append({
            "participant_id": person.participant_id, "site_id": person.site_id,
            "dx_group": person.dx_group, "age": person.age, "sex": person.sex,
            "processed": int(entry.processed),
            "run_stem": run.stem if run else None,
            "bold_key": run.bold_key if run else None,
            "confounds_key": run.confounds_key if run else None,
            "bold_bytes": run.bold_bytes if run else None,
            "exclusion": entry.exclusion,
        })
    _write_csv(args.data_dir / COHORT_CSV, COHORT_FIELDS, rows)
    (args.data_dir / SITE_TR_JSON).write_text(json.dumps(site_tr, indent=2, sort_keys=True))

    eligible = [r for r in rows if not r["exclusion"]]
    n_asd = sum(1 for r in eligible if r["dx_group"] == 1)
    n_tc = sum(1 for r in eligible if r["dx_group"] == 2)
    gb = sum(int(r["bold_bytes"] or 0) for r in eligible) / 1e9
    print(f"\n  participants (all sites) : {len(rows)}")
    print(f"  with fMRIPrep output     : {sum(r['processed'] for r in rows)}")
    print(f"  eligible for extraction  : {len(eligible)}  (ASD {n_asd}, TC {n_tc})")
    print(f"  BOLD to stream           : {gb:.0f} GB")
    reasons: Dict[str, int] = {}
    for r in rows:
        if r["exclusion"]:
            reasons[r["exclusion"]] = reasons.get(r["exclusion"], 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"  excluded — {reason}: {count}")
    print(f"\nWrote {args.data_dir / COHORT_CSV}")
    return 0


# ── extract ────────────────────────────────────────────────────────────────

def _run_from_row(row: Dict[str, str]) -> abide2.RestRun:
    return abide2.RestRun(
        participant_id=row["participant_id"], session="", run="",
        stem=row["run_stem"], bold_key=row["bold_key"],
        confounds_key=row["confounds_key"], bold_bytes=int(row["bold_bytes"] or 0),
    )


def cmd_extract(args: argparse.Namespace) -> int:
    cohort_path = args.data_dir / COHORT_CSV
    if not cohort_path.exists():
        print(f"{cohort_path} not found — run `discover` first.", file=sys.stderr)
        return 2
    cohort = _read_csv(cohort_path)
    site_tr = json.loads((args.data_dir / SITE_TR_JSON).read_text())

    extraction_path = args.data_dir / EXTRACTION_CSV
    done: Dict[str, Dict[str, object]] = {}
    if extraction_path.exists():
        for row in _read_csv(extraction_path):
            if not row["error"]:  # errors are retried; outcomes are kept
                done[row["participant_id"]] = row

    todo = sorted(
        (r for r in cohort if not r["exclusion"] and r["participant_id"] not in done),
        key=lambda r: r["participant_id"],
    )
    if args.n_subjects:
        todo = todo[: args.n_subjects]
    print(f"{len(done)} already extracted; extracting {len(todo)} "
          f"with {args.workers} workers...")

    roi_dir = args.data_dir / ROI_DIR
    work_dir = args.data_dir / WORK_DIR
    atlas = abide2.load_atlas(args.data_dir)

    def _one(row: Dict[str, str]) -> Dict[str, object]:
        try:
            result = abide2.extract_participant(
                _run_from_row(row), site_tr.get(row["site_id"]), atlas,
                roi_dir, work_dir, keep_bold=args.keep_bold,
            )
            out = asdict(result)
            out["path"] = str(result.path) if result.path else None
            out["error"] = None
            return out
        except Exception as exc:  # noqa: BLE001 - one bad download must not end the run
            return {"participant_id": row["participant_id"],
                    "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, row) for row in todo]
        for i, future in enumerate(as_completed(futures), 1):
            out = future.result()
            done[str(out["participant_id"])] = out
            status = out.get("error") or out.get("exclusion") or "ok"
            print(f"  [{i}/{len(todo)}] {out['participant_id']}: {status}")
            if i % 10 == 0 or i == len(todo):
                _write_csv(extraction_path, EXTRACTION_FIELDS,
                           [done[k] for k in sorted(done)])

    _write_csv(extraction_path, EXTRACTION_FIELDS, [done[k] for k in sorted(done)])
    errors = [r for r in done.values() if r.get("error")]
    ok = [r for r in done.values() if r.get("path")]
    print(f"\n{len(ok)} .1D files in {roi_dir}; {len(errors)} operational errors.")
    if errors:
        print("Re-run `extract` to retry errors; `evaluate` refuses an incomplete set.")
    return 0


# ── evaluate ───────────────────────────────────────────────────────────────

def _completeness(cohort: List[Dict[str, str]], extraction: Dict[str, Dict[str, str]]) -> List[str]:
    """Eligible participants with neither a .1D nor a preregistered exclusion."""
    missing = []
    for row in cohort:
        if row["exclusion"]:
            continue
        out = extraction.get(row["participant_id"])
        if out is None or out.get("error") or not (out.get("path") or out.get("exclusion")):
            missing.append(row["participant_id"])
    return missing


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        frozen = require_frozen(args.prereg_sha, _REPO)
    except PreregistrationNotFrozen as exc:
        print(f"Refusing to evaluate: {exc}", file=sys.stderr)
        return 3

    cohort = _read_csv(args.data_dir / COHORT_CSV)
    extraction = {r["participant_id"]: r for r in _read_csv(args.data_dir / EXTRACTION_CSV)}
    missing = _completeness(cohort, extraction)
    if missing and not args.allow_incomplete:
        print(f"Refusing to evaluate: {len(missing)} eligible participants have no "
              f"outcome (e.g. {missing[:5]}). Re-run `extract`, or pass "
              "--allow-incomplete (recorded in the metrics).", file=sys.stderr)
        return 4

    from predict import _load_ensemble, predict_file  # torch import deferred to here

    models = _load_ensemble(args.device)
    print(f"Loaded {len(models)} checkpoints. Scoring...")

    by_id = {r["participant_id"]: r for r in cohort}
    predictions = []
    for pid in sorted(extraction):
        out = extraction[pid]
        if not out.get("path") or out.get("error"):
            continue
        result = predict_file(Path(out["path"]), models, args.device)
        row = by_id[pid]
        predictions.append({
            "participant_id": pid, "site_id": row["site_id"],
            "dx_group": int(row["dx_group"]), "age": _float(row["age"]),
            "mean_fd": _float(out.get("mean_fd", "")), "p_asd": result["p_asd"],
            **{f"p_{site.lower()}": p for site, p in result["per_model"]},
        })

    dx = [p["dx_group"] for p in predictions]
    primary = analysis.primary_analysis(
        dx, [p["p_asd"] for p in predictions],
        [p["site_id"] for p in predictions], [p["age"] for p in predictions],
    )

    included_ids = {p["participant_id"] for p in predictions}

    def _bias(rows: List[Dict[str, str]]) -> Dict[str, object]:
        return analysis.selection_bias(
            included=np.array([r["participant_id"] in included_ids for r in rows]),
            dx_group=np.array([_float(r["dx_group"]) for r in rows]),
            ages=np.array([_float(r["age"]) for r in rows]),
            sex=np.array([_float(r["sex"]) for r in rows]),
            sites=np.array([r["site_id"] for r in rows]),
        )

    labeled = [r for r in cohort if r["dx_group"] in ("1", "2")]
    metrics = {
        "prereg_sha": frozen,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "allow_incomplete": bool(args.allow_incomplete),
        "n_missing_outcomes": len(missing),
        "primary": primary,
        "c2_selection_bias": {
            "included_vs_processed_but_excluded": _bias(
                [r for r in labeled if r["processed"] == "1"]),
            "included_vs_all_labeled": _bias(labeled),
            "mean_fd_by_class": analysis.mean_fd_by_class(
                dx, [p["mean_fd"] for p in predictions]),
        },
        "c1_pipeline_control": "not computed by this script; see preregistration §7 C1",
        "arm_b": "not computed by this script; see preregistration §8",
    }

    fields = list(predictions[0].keys()) if predictions else ["participant_id"]
    _write_csv(args.data_dir / PREDICTIONS_CSV, fields, predictions)
    (args.data_dir / METRICS_JSON).write_text(json.dumps(metrics, indent=2, default=str))

    print(f"\nVerdict: {primary['verdict']}  (endpoint: {primary.get('endpoint')})")
    if "pooled" in primary:
        pooled = primary["pooled"]
        print(f"Pooled AUC {pooled['auc']:.4f}  95% CI [{pooled['ci_lower']:.4f}, "
              f"{pooled['ci_upper']:.4f}]  vs baseline {analysis.BASELINE_AUC}")
    print(f"Wrote {args.data_dir / METRICS_JSON}. Record the verdict in the "
          "preregistration's §10 results ledger before starting Arm B.")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot ABIDE II evaluation of the ABIDE I CC200 ensemble "
                    "(preregistered; see preregistrations/).")
    parser.add_argument("--data-dir", type=Path, default=_REPO / "data" / "abide2",
                        help="Working directory for manifests, .1D files and results "
                             "(default: data/abide2, git-ignored)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("discover", help="Build the cohort from listings and phenotype files")

    extract = sub.add_parser("extract", help="Stream BOLD and write CC200 .1D files")
    extract.add_argument("--n-subjects", type=int, default=None,
                         help="Only the first N remaining participants (smoke test)")
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--keep-bold", action="store_true",
                         help="Keep BOLD files after extraction (~170 GB for the eligible cohort)")

    evaluate = sub.add_parser("evaluate", help="Score .1D files and apply the analysis")
    evaluate.add_argument("--prereg-sha", default=None,
                          help="Frozen preregistration commit (required)")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--allow-incomplete", action="store_true",
                          help="Evaluate despite missing outcomes (recorded in metrics)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"discover": cmd_discover, "extract": cmd_extract,
               "evaluate": cmd_evaluate}[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
