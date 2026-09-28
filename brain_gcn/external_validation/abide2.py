"""
ABIDE II inputs for zero-shot evaluation, built on the dataset's real S3 layout.

ABIDE II was never released as C-PAC ROI time series: the path the original
eval_abide2.py requested is empty, and its phenotype URL returns 404. What does
exist is fMRIPrep output for 732 participants plus per-site BIDS phenotype
files. This module turns those into .1D files that approximate the ABIDE I
C-PAC ``filt_global`` CC200 inputs the model was trained on.

Every processing choice below implements §4 of
preregistrations/2026-09-28-abide2-external-validation.md. Change them only
through that document's deviations log.

S3 layout (public fcp-indi bucket, verified 2026-09-28):
  ABIDE2/Outputs/fmriprep/fmriprep/sub-<ID>/[ses-<S>/]func/
      sub-<ID>_[ses-<S>_]task-rest_[run-<R>_]space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
      sub-<ID>_[ses-<S>_]task-rest_[run-<R>_]desc-confounds_regressors.tsv
  ABIDE2/RawData/<SITE>/participants.tsv       (labels: dx_group 1=ASD, 2=TC)
  ABIDE2/RawData/<SITE>/task-rest_bold.json    (RepetitionTime)
  ABIDE_Initiative/Resources/cc200_roi_atlas.nii.gz   (the atlas ABIDE I used)
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

S3_BASE = "https://s3.amazonaws.com/fcp-indi"
FMRIPREP_PREFIX = "data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/"
RAWDATA_PREFIX = "data/Projects/ABIDE2/RawData/"
CC200_ATLAS_KEY = "data/Projects/ABIDE_Initiative/Resources/cc200_roi_atlas.nii.gz"

BOLD_SUFFIX = "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
T1W_MNI_SUFFIX = "_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
# Newer fMRIPrep writes *_timeseries.tsv; the ABIDE II release predates that.
CONFOUND_SUFFIXES = ("_desc-confounds_timeseries.tsv", "_desc-confounds_regressors.tsv")

N_ROIS = 200
# predict.py rejects series shorter than _WINDOW_LEN (50) + _STEP (3).
MIN_VOLUMES = 53
BANDPASS_HZ = (0.01, 0.1)
N_ACOMPCOR = 5
TR_TOLERANCE = 0.01
MOTION_COLUMNS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
ACOMPCOR_COLUMNS = tuple(f"a_comp_cor_{i:02d}" for i in range(N_ACOMPCOR))
GLOBAL_SIGNAL_COLUMN = "global_signal"

Fetch = Callable[[str], bytes]


# ── transport ──────────────────────────────────────────────────────────────

def http_get(url: str, retries: int = 3, timeout: float = 120.0) -> bytes:
    """GET with bounded retries. Honours HTTPS_PROXY and SSL_CERT_FILE."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError:
            raise  # 403/404 are answers, not transient failures
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def download(url: str, dest: Path, fetch: Fetch = http_get) -> Path:
    """Write ``url`` to ``dest`` atomically; an existing ``dest`` is reused."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(fetch(url))
    tmp.replace(dest)
    return dest


def object_url(key: str) -> str:
    return f"{S3_BASE}/{key}"


# ── S3 listing ─────────────────────────────────────────────────────────────

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_list_response(
    xml_text: bytes,
) -> Tuple[List[Tuple[str, int]], List[str], Optional[str]]:
    """Parse one ListObjectsV2 page into (keys+sizes, common prefixes, next token)."""
    root = ET.fromstring(xml_text)
    objects: List[Tuple[str, int]] = []
    prefixes: List[str] = []
    token: Optional[str] = None
    for child in root:
        name = _local(child.tag)
        if name == "Contents":
            fields = {_local(c.tag): (c.text or "") for c in child}
            objects.append((fields.get("Key", ""), int(fields.get("Size", "0") or 0)))
        elif name == "CommonPrefixes":
            for c in child:
                if _local(c.tag) == "Prefix" and c.text:
                    prefixes.append(c.text)
        elif name == "NextContinuationToken":
            token = child.text or None
    return objects, prefixes, token


def list_objects(
    prefix: str, fetch: Fetch = http_get, delimiter: Optional[str] = None,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """All objects (and common prefixes) under ``prefix``, following pagination."""
    objects: List[Tuple[str, int]] = []
    prefixes: List[str] = []
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        page_objects, page_prefixes, token = parse_list_response(
            fetch(f"{S3_BASE}?{urllib.parse.urlencode(params)}")
        )
        objects.extend(page_objects)
        prefixes.extend(page_prefixes)
        if not token:
            return objects, prefixes


# ── run selection (§4.1 rules 2-3) ─────────────────────────────────────────

@dataclass(frozen=True)
class RestRun:
    participant_id: str
    session: str
    run: str
    stem: str
    bold_key: str
    confounds_key: str
    bold_bytes: int


_ENTITY_RE = re.compile(r"(?:^|_)(sub|ses|task|acq|run)-([A-Za-z0-9]+)")


def _entities(stem: str) -> Dict[str, str]:
    return dict(_ENTITY_RE.findall(stem))


def _natural(value: str) -> Tuple[int, int, str]:
    """Order absent < numeric (by value) < other strings."""
    if not value:
        return (0, 0, "")
    if value.isdigit():
        return (1, int(value), "")
    return (2, 0, value)


def select_runs(objects: Iterable[Tuple[str, int]]) -> Dict[str, RestRun]:
    """One resting-state run per participant: lowest session, then lowest run.

    Only runs with both an MNI152NLin2009cAsym preprocessed BOLD and a confounds
    file for the same run are eligible.
    """
    bold: Dict[Tuple[str, str], Tuple[str, int]] = {}
    confounds: Dict[Tuple[str, str], str] = {}
    for key, size in objects:
        name = key.rsplit("/", 1)[-1]
        if "task-rest" not in name:
            continue
        if name.endswith(BOLD_SUFFIX):
            stem = name[: -len(BOLD_SUFFIX)]
            pid = _entities(stem).get("sub")
            if pid:
                bold[(pid, stem)] = (key, size)
            continue
        for suffix in CONFOUND_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                pid = _entities(stem).get("sub")
                if pid:
                    confounds.setdefault((pid, stem), key)
                break

    candidates: Dict[str, List[RestRun]] = {}
    for (pid, stem), (bold_key, size) in bold.items():
        conf_key = confounds.get((pid, stem))
        if conf_key is None:
            continue
        ents = _entities(stem)
        candidates.setdefault(pid, []).append(RestRun(
            participant_id=pid,
            session=ents.get("ses", ""),
            run=ents.get("run", ""),
            stem=stem,
            bold_key=bold_key,
            confounds_key=conf_key,
            bold_bytes=size,
        ))

    return {
        pid: min(runs, key=lambda r: (_natural(r.session), _natural(r.run), r.stem))
        for pid, runs in candidates.items()
    }


def processed_participants(objects: Iterable[Tuple[str, int]]) -> set:
    """Participant IDs that have any fMRIPrep output folder."""
    found = set()
    for key, _ in objects:
        match = re.search(r"/sub-([A-Za-z0-9]+)/", key)
        if match:
            found.add(match.group(1))
    return found


# ── phenotype (§4.1 rule 1) ────────────────────────────────────────────────

@dataclass(frozen=True)
class Participant:
    participant_id: str
    site_id: str
    dx_group: Optional[int]
    age: Optional[float]
    sex: Optional[int]


def _normalize_id(raw: str) -> str:
    value = str(raw).strip()
    return value[4:] if value.startswith("sub-") else value


def _maybe_int(raw: object, allowed: Optional[Tuple[int, ...]] = None) -> Optional[int]:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    if allowed is not None and value not in allowed:
        return None
    return value


def _maybe_float(raw: object) -> Optional[float]:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def _decode(raw: bytes) -> str:
    """UTF-8 (BOM tolerated), falling back to Latin-1.

    ABIDEII-GU_1's participants.tsv holds one Latin-1 non-breaking space, in a
    free-text medication field; Latin-1 decodes every byte, and ``strip()``
    removes the NBSP from any field that is parsed.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_participants(tsv: bytes, site_folder: str) -> List[Participant]:
    """Parse one site's BIDS participants.tsv.

    The real files carry a column named ``'age_at_scan '`` with a trailing
    space, so headers are stripped before lookup.
    """
    frame = pd.read_csv(io.StringIO(_decode(tsv)), sep="\t", dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    rows: List[Participant] = []
    for record in frame.to_dict("records"):
        pid = _normalize_id(record.get("participant_id", ""))
        if not pid:
            continue
        rows.append(Participant(
            participant_id=pid,
            site_id=(record.get("site_id") or site_folder).strip(),
            dx_group=_maybe_int(record.get("dx_group"), allowed=(1, 2)),
            age=_maybe_float(record.get("age_at_scan")),
            sex=_maybe_int(record.get("sex")),
        ))
    return rows


def parse_repetition_time(sidecar: bytes) -> Optional[float]:
    try:
        value = json.loads(sidecar).get("RepetitionTime")
    except (ValueError, AttributeError):
        return None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


# ── discovery ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CohortEntry:
    participant: Participant
    run: Optional[RestRun]
    processed: bool
    exclusion: Optional[str]

    @property
    def eligible_for_extraction(self) -> bool:
        return self.exclusion is None


EXCLUDE_DX = "rule 1: dx_group missing or not 1/2"
EXCLUDE_NOT_LISTED = "rule 1: not listed in any participants.tsv"
EXCLUDE_NO_SITE_LABELS = "rule 1: site has no participants.tsv in the public bucket"
SITE_FOLDER_PREFIX = "ABIDEII-"  # RawData also holds non-site folders, e.g. 'sidecards'
EXCLUDE_NOT_PROCESSED = "rule 2: no fMRIPrep output"
EXCLUDE_NO_BOLD = "rule 2: no MNI preprocessed BOLD with confounds"
EXCLUDE_REGRESSORS = "rule 2: confounds lack required regressors"
EXCLUDE_SHORT = f"rule 4: fewer than {MIN_VOLUMES} volumes"
EXCLUDE_TR = "rule 5: header TR disagrees with site RepetitionTime"
EXCLUDE_ROWS = "rule 6: confounds rows != BOLD volumes"


def build_cohort(
    participants: Iterable[Participant],
    runs: Dict[str, RestRun],
    processed: set,
    unlabeled: Optional[Dict[str, str]] = None,
) -> List[CohortEntry]:
    """Apply the pre-download inclusion rules (§4.1 rules 1-3).

    Rules 4-6 need the BOLD header and confounds file, so they are applied
    during extraction. Every participant is kept in the result, with the first
    rule they fail, because C2 compares included and excluded participants.

    ``unlabeled`` maps participant ID to site for sites whose folder has
    imaging but no participants.tsv; they are excluded under rule 1 with their
    site recorded, rather than disappearing into "not listed".
    """
    entries: List[CohortEntry] = []
    seen = set()
    for person in participants:
        if person.participant_id in seen:
            log.warning("duplicate participant %s ignored", person.participant_id)
            continue
        seen.add(person.participant_id)
        was_processed = person.participant_id in processed
        run = runs.get(person.participant_id)
        if person.dx_group is None:
            reason: Optional[str] = EXCLUDE_DX
        elif not was_processed:
            reason = EXCLUDE_NOT_PROCESSED
        elif run is None:
            reason = EXCLUDE_NO_BOLD
        else:
            reason = None
        entries.append(CohortEntry(person, run, was_processed, reason))

    for pid, site in sorted((unlabeled or {}).items()):
        if pid in seen:
            continue
        seen.add(pid)
        stub = Participant(pid, site_id=site, dx_group=None, age=None, sex=None)
        entries.append(CohortEntry(stub, runs.get(pid), pid in processed,
                                   EXCLUDE_NO_SITE_LABELS))

    for pid in sorted(processed - seen):
        stub = Participant(pid, site_id="", dx_group=None, age=None, sex=None)
        entries.append(CohortEntry(stub, runs.get(pid), True, EXCLUDE_NOT_LISTED))
    return entries


def discover(fetch: Fetch = http_get) -> Tuple[List[CohortEntry], Dict[str, Optional[float]]]:
    """Build the cohort from listings and phenotype files only.

    Reads no imaging data and loads no model, so it is safe to run before the
    preregistration is frozen. Returns the cohort and each site's TR.
    """
    objects, _ = list_objects(FMRIPREP_PREFIX, fetch)
    runs = select_runs(objects)
    processed = processed_participants(objects)

    _, site_prefixes = list_objects(RAWDATA_PREFIX, fetch, delimiter="/")
    participants: List[Participant] = []
    unlabeled: Dict[str, str] = {}
    site_tr: Dict[str, Optional[float]] = {}
    for prefix in sorted(site_prefixes):
        site = prefix.rstrip("/").rsplit("/", 1)[-1]
        if not site.startswith(SITE_FOLDER_PREFIX):
            continue
        try:
            site_tr[site] = parse_repetition_time(fetch(object_url(prefix + "task-rest_bold.json")))
        except urllib.error.HTTPError:
            site_tr[site] = None
        try:
            participants.extend(parse_participants(fetch(object_url(prefix + "participants.tsv")), site))
        except urllib.error.HTTPError:
            _, subject_prefixes = list_objects(prefix, fetch, delimiter="/")
            ids = [p.rstrip("/").rsplit("/", 1)[-1] for p in subject_prefixes]
            ids = [i[4:] for i in ids if i.startswith("sub-")]
            log.warning("site %s has no participants.tsv; %d participants unlabeled", site, len(ids))
            unlabeled.update({pid: site for pid in ids})

    return build_cohort(participants, runs, processed, unlabeled), site_tr


# ── signal processing (§4.2) ───────────────────────────────────────────────

def friston24(motion: np.ndarray) -> np.ndarray:
    """(T, 6) rigid-body parameters -> (T, 24).

    Columns: the parameters, their backward differences (first row 0), and the
    squares of both. Computed here because the ABIDE II fMRIPrep release does
    not ship the expansion.
    """
    diff = np.vstack([np.zeros((1, motion.shape[1])), np.diff(motion, axis=0)])
    return np.hstack([motion, diff, motion ** 2, diff ** 2])


def nuisance_design(confounds: pd.DataFrame) -> np.ndarray:
    """Regressors approximating C-PAC ``filt_global``.

    Intercept, linear and quadratic trends, Friston-24 motion, aCompCor 00-04
    and the global signal: 33 columns. Raises ValueError naming any missing
    regressor.
    """
    required = MOTION_COLUMNS + ACOMPCOR_COLUMNS + (GLOBAL_SIGNAL_COLUMN,)
    missing = [c for c in required if c not in confounds.columns]
    if missing:
        raise ValueError(f"confounds file lacks required regressors: {missing}")
    n = len(confounds)
    t = np.linspace(-1.0, 1.0, n)
    trends = np.column_stack([np.ones(n), t, t ** 2])
    values = confounds[list(required)].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    motion = values[:, : len(MOTION_COLUMNS)]
    rest = values[:, len(MOTION_COLUMNS):]
    return np.hstack([trends, friston24(motion), rest])


def regress_out(signals: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residuals of an ordinary least-squares fit of ``signals`` on ``design``."""
    beta, *_ = np.linalg.lstsq(design, signals, rcond=None)
    return signals - design @ beta


def ideal_bandpass(signals: np.ndarray, tr: float, low: float, high: float) -> np.ndarray:
    """Zero every frequency outside [low, high] Hz.

    Zero-pads to the next power of two before the FFT to limit wrap-around, as
    C-PAC's ideal filter also pads. Exact padding length and cutoff rounding
    may differ from C-PAC; that difference belongs to the pipeline shift the
    preregistration's control C1 measures.
    """
    n = signals.shape[0]
    padded = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(signals, n=padded, axis=0)
    freqs = np.fft.rfftfreq(padded, d=tr)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, n=padded, axis=0)[:n]


def header_tr(img) -> Optional[float]:
    """Repetition time in seconds from a 4D NIfTI header, or None."""
    zooms = img.header.get_zooms()
    if len(zooms) < 4 or not zooms[3] or zooms[3] <= 0:
        return None
    tr = float(zooms[3])
    unit = img.header.get_xyzt_units()[1]
    if unit == "msec":
        tr /= 1000.0
    elif unit == "usec":
        tr /= 1e6
    return tr


def roi_means(bold_img, atlas_img, n_rois: int = N_ROIS) -> np.ndarray:
    """(T, n_rois) mean signal per atlas label, column i holding label i + 1.

    The atlas is resampled to the BOLD grid by nearest neighbour. A label with
    no voxels on that grid yields a zero column rather than being dropped, so
    column positions always match the ABIDE I ``#1 ... #200`` header.
    """
    import nibabel as nib
    from nilearn.image import resample_to_img
    from scipy import sparse

    shape3d = bold_img.shape[:3]
    reference = nib.Nifti1Image(np.zeros(shape3d, dtype=np.int8), bold_img.affine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nilearn API-transition FutureWarnings
        atlas_rs = resample_to_img(atlas_img, reference, interpolation="nearest")
    labels = np.rint(np.asarray(atlas_rs.dataobj)).astype(np.int64).ravel()

    data = bold_img.get_fdata(dtype=np.float32)
    n_volumes = data.shape[3]
    flat = data.reshape(-1, n_volumes)
    inside = (labels >= 1) & (labels <= n_rois)
    idx = labels[inside] - 1
    membership = sparse.csr_matrix(
        (np.ones(idx.size, dtype=np.float64), (idx, np.arange(idx.size))),
        shape=(n_rois, idx.size),
    )
    sums = membership @ flat[inside].astype(np.float64)
    counts = np.bincount(idx, minlength=n_rois).astype(np.float64)[:, None]
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means.T


def write_1d(timeseries: np.ndarray, path: Path) -> Path:
    """Write ABIDE I-format .1D: '#1\\t...\\t#N' header, tab-separated, 6 decimals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(f"#{i}" for i in range(1, timeseries.shape[1] + 1))
    np.savetxt(path, timeseries, fmt="%.6f", delimiter="\t", header=header, comments="")
    return path


# ── extraction (§4.1 rules 4-6, §4.2) ──────────────────────────────────────

@dataclass(frozen=True)
class Extraction:
    participant_id: str
    path: Optional[Path]
    exclusion: Optional[str]
    n_volumes: Optional[int] = None
    tr: Optional[float] = None
    site_tr: Optional[float] = None
    n_empty_rois: Optional[int] = None
    mean_fd: Optional[float] = None


def output_path(out_dir: Path, participant_id: str) -> Path:
    return out_dir / f"{participant_id}_rois_cc200.1D"


def extract_participant(
    run: RestRun,
    site_tr: Optional[float],
    atlas_img,
    out_dir: Path,
    work_dir: Path,
    fetch: Fetch = http_get,
    keep_bold: bool = False,
) -> Extraction:
    """Download one run, apply §4.1 rules 4-6, and write its CC200 .1D.

    The confounds file is fetched first so a participant excluded on its
    regressors never costs a ~220 MB BOLD download. The BOLD file is deleted
    after extraction unless ``keep_bold``: the eligible cohort is ~170 GB.
    """
    import nibabel as nib

    pid = run.participant_id
    conf_path = download(object_url(run.confounds_key),
                         work_dir / run.confounds_key.rsplit("/", 1)[-1], fetch)
    confounds = pd.read_csv(conf_path, sep="\t", na_values=["n/a"])
    try:
        design = nuisance_design(confounds)
    except ValueError:
        return Extraction(pid, None, EXCLUDE_REGRESSORS, site_tr=site_tr)

    bold_path = work_dir / run.bold_key.rsplit("/", 1)[-1]
    try:
        download(object_url(run.bold_key), bold_path, fetch)
        img = nib.load(str(bold_path))
        n_volumes = int(img.shape[3]) if len(img.shape) == 4 else 0
        tr = header_tr(img)
        if n_volumes < MIN_VOLUMES:
            return Extraction(pid, None, EXCLUDE_SHORT, n_volumes, tr, site_tr)
        if tr is None or (site_tr is not None and abs(tr - site_tr) > TR_TOLERANCE * site_tr):
            return Extraction(pid, None, EXCLUDE_TR, n_volumes, tr, site_tr)
        if len(confounds) != n_volumes:
            return Extraction(pid, None, EXCLUDE_ROWS, n_volumes, tr, site_tr)

        raw = roi_means(img, atlas_img)
        cleaned = ideal_bandpass(regress_out(raw, design), tr, *BANDPASS_HZ)
        n_empty = int((raw.std(axis=0) < 1e-8).sum())
        fd = confounds.get("framewise_displacement")
        mean_fd = float(np.nanmean(fd)) if fd is not None and fd.notna().any() else None
        path = write_1d(cleaned, output_path(out_dir, pid))
        return Extraction(pid, path, None, n_volumes, tr, site_tr, n_empty, mean_fd)
    finally:
        if not keep_bold and bold_path.exists():
            bold_path.unlink()


def load_atlas(cache_dir: Path, fetch: Fetch = http_get):
    """The CC200 atlas file ABIDE I used, downloaded once and cached."""
    import nibabel as nib

    path = download(object_url(CC200_ATLAS_KEY), cache_dir / "cc200_roi_atlas.nii.gz", fetch)
    return nib.load(str(path))
