"""
Preregistered analysis for the ABIDE II external validation.

Implements §6 (primary endpoint and decision rule) and §7 (controls C2-C4) of
preregistrations/2026-09-28-abide2-external-validation.md. The constants below
are the plan's [PROPOSED] values; after the freeze they change only through the
document's deviations log.

Labels follow ABIDE: dx_group 1 = ASD (positive class), 2 = typical control.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

BASELINE_AUC = 0.7298          # 20-site ABIDE I leave-one-site-out, N=1,102
NONINFERIORITY_MARGIN = 0.05
REPLICATED_CI_FLOOR = 0.60
CHANCE_AUC = 0.5
N_BOOTSTRAP = 2000
SEED = 20260928
OPERATING_THRESHOLD = 0.5
MIN_PER_CLASS_OVERALL = 100    # kill condition K5
MIN_PER_CLASS_SITE = 10        # C4
MIN_PER_CLASS_AGE_BAND = 20    # C3
SITE_CONFOUND_TOLERANCE = 0.05 # C4
SELECTION_BIAS_ALPHA = 0.05    # C2
AGE_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 12.0), (12.0, 18.0), (18.0, 30.0), (30.0, float("inf")),
)

REPLICATED = "REPLICATED"
ATTENUATED = "ATTENUATED"
FAILED = "FAILED"
UNDERPOWERED = "UNDERPOWERED"


def labels_from_dx(dx_group: Sequence[int]) -> np.ndarray:
    """dx_group 1/2 -> 1 (ASD) / 0 (TC). Anything else is a caller error."""
    dx = np.asarray(dx_group)
    if not np.isin(dx, (1, 2)).all():
        raise ValueError("dx_group must contain only 1 (ASD) and 2 (TC)")
    return (dx == 1).astype(int)


def _class_counts(y: np.ndarray) -> Tuple[int, int]:
    return int((y == 1).sum()), int((y == 0).sum())


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    statistic,
    strata: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Tuple[float, float]:
    """Percentile 95% CI, resampling with replacement within each stratum.

    Stratifying by class (and by site x class for C4) keeps every resample's
    composition equal to the observed one, so no resample loses a class.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        values[i] = statistic(idx)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def pooled_auc_with_ci(y: np.ndarray, scores: np.ndarray, **kwargs) -> Dict[str, float]:
    """§6 primary endpoint: one AUC over all participants, class-stratified CI."""
    lower, upper = bootstrap_ci(
        y, scores, lambda idx: roc_auc_score(y[idx], scores[idx]), strata=y, **kwargs,
    )
    return {"auc": float(roc_auc_score(y, scores)), "ci_lower": lower, "ci_upper": upper}


def verdict(
    auc: float,
    ci_lower: float,
    baseline: float = BASELINE_AUC,
    margin: float = NONINFERIORITY_MARGIN,
    ci_floor: float = REPLICATED_CI_FLOOR,
    chance: float = CHANCE_AUC,
) -> str:
    """§6 decision rule. Exhaustive and mutually exclusive by construction."""
    if ci_lower <= chance:
        return FAILED
    # Rounded so an edited baseline cannot move the boundary by float error
    # (0.73 - 0.05 is 0.6799999999999999).
    if auc >= round(baseline - margin, 10) and ci_lower > ci_floor:
        return REPLICATED
    return ATTENUATED


def operating_point(y: np.ndarray, scores: np.ndarray,
                    threshold: float = OPERATING_THRESHOLD) -> Dict[str, float]:
    """Accuracy, sensitivity and specificity at the fixed, untuned threshold."""
    predicted = (scores > threshold).astype(int)
    n_pos, n_neg = _class_counts(y)
    return {
        "threshold": threshold,
        "accuracy": float((predicted == y).mean()),
        "sensitivity": float(((predicted == 1) & (y == 1)).sum() / n_pos) if n_pos else float("nan"),
        "specificity": float(((predicted == 0) & (y == 0)).sum() / n_neg) if n_neg else float("nan"),
    }


def per_site_auc(
    y: np.ndarray, scores: np.ndarray, sites: np.ndarray,
    min_per_class: int = MIN_PER_CLASS_SITE,
) -> List[Dict[str, object]]:
    """Per-site AUC; sites below ``min_per_class`` in either class get counts only."""
    rows: List[Dict[str, object]] = []
    for site in sorted(np.unique(sites)):
        mask = sites == site
        n_asd, n_tc = _class_counts(y[mask])
        auc: Optional[float] = None
        if min(n_asd, n_tc) >= min_per_class:
            auc = float(roc_auc_score(y[mask], scores[mask]))
        rows.append({"site": str(site), "n_asd": n_asd, "n_tc": n_tc, "auc": auc})
    return rows


def _weighted_within_site(y: np.ndarray, scores: np.ndarray, sites: np.ndarray,
                          min_per_class: int) -> float:
    total, weight = 0.0, 0
    for site in np.unique(sites):
        mask = sites == site
        n_asd, n_tc = _class_counts(y[mask])
        if min(n_asd, n_tc) < min_per_class:
            continue
        n = int(mask.sum())
        total += n * roc_auc_score(y[mask], scores[mask])
        weight += n
    return total / weight if weight else float("nan")


def site_confound(
    y: np.ndarray, scores: np.ndarray, sites: np.ndarray, pooled_auc: float,
    min_per_class: int = MIN_PER_CLASS_SITE,
    tolerance: float = SITE_CONFOUND_TOLERANCE,
    **kwargs,
) -> Dict[str, object]:
    """C4: within-site mean AUC (n-weighted) against the pooled AUC.

    A model that has learned to recognise sites can score a high pooled AUC
    when ASD prevalence differs between sites, while separating nobody within
    any site. If pooled exceeds the within-site mean by more than
    ``tolerance``, the verdict moves to the within-site mean.
    """
    within = _weighted_within_site(y, scores, sites, min_per_class)
    strata = np.array([f"{s}|{c}" for s, c in zip(sites, y)])
    lower, upper = bootstrap_ci(
        y, scores,
        lambda idx: _weighted_within_site(y[idx], scores[idx], sites[idx], min_per_class),
        strata=strata, **kwargs,
    )
    flagged = bool(np.isfinite(within) and pooled_auc - within > tolerance)
    return {"within_site_mean_auc": within, "ci_lower": lower, "ci_upper": upper,
            "pooled_minus_within": pooled_auc - within, "flagged": flagged}


def age_bands(
    y: np.ndarray, scores: np.ndarray, ages: np.ndarray,
    bands: Sequence[Tuple[float, float]] = AGE_BANDS,
    min_per_class: int = MIN_PER_CLASS_AGE_BAND,
    **kwargs,
) -> List[Dict[str, object]]:
    """C3: AUC per fixed age band; under-filled bands report counts only."""
    rows: List[Dict[str, object]] = []
    for low, high in bands:
        mask = np.isfinite(ages) & (ages >= low) & (ages < high)
        n_asd, n_tc = _class_counts(y[mask])
        row: Dict[str, object] = {"band": [low, high], "n_asd": n_asd, "n_tc": n_tc,
                                  "auc": None, "ci_lower": None, "ci_upper": None}
        if min(n_asd, n_tc) >= min_per_class:
            row.update(pooled_auc_with_ci(y[mask], scores[mask], **kwargs))
        rows.append(row)
    return rows


def _chi2_p(included: np.ndarray, values: np.ndarray, present: np.ndarray) -> Optional[float]:
    """Chi-square p for category x inclusion over rows where ``present``."""
    from scipy.stats import chi2_contingency

    if not present.any():
        return None
    inc, vals = included[present], values[present]
    table = np.array([[np.sum(inc & (vals == v)), np.sum(~inc & (vals == v))]
                      for v in np.unique(vals)])
    if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
        return None
    return float(chi2_contingency(table)[1])


def selection_bias(
    included: np.ndarray, dx_group: np.ndarray, ages: np.ndarray, sex: np.ndarray,
    sites: np.ndarray,
) -> Dict[str, object]:
    """C2: does inclusion depend on diagnosis, age, sex, or site?

    ``included`` marks participants who reached the model; everyone else in the
    arrays is the comparison group. Rows missing a variable are dropped from
    that variable's test only (missing numbers are NaN, missing sites are "").
    """
    from scipy.stats import mannwhitneyu

    dx = dx_group.astype(float)
    sex_f = sex.astype(float)
    ok_age = np.isfinite(ages)
    age_p: Optional[float] = None
    if (included & ok_age).any() and (~included & ok_age).any():
        age_p = float(mannwhitneyu(ages[included & ok_age], ages[~included & ok_age]).pvalue)

    site_arr = np.asarray(sites, dtype=str)
    dx_p = _chi2_p(included, dx, np.isfinite(dx))
    return {
        "n_included": int(included.sum()),
        "n_excluded": int((~included).sum()),
        "dx_by_inclusion_p": dx_p,
        "age_by_inclusion_p": age_p,
        "sex_by_inclusion_p": _chi2_p(included, sex_f, np.isfinite(sex_f)),
        "site_by_inclusion_p": _chi2_p(included, site_arr, site_arr != ""),
        "conditional_on_preprocessing": bool(dx_p is not None and dx_p < SELECTION_BIAS_ALPHA),
    }


def mean_fd_by_class(dx_group: Sequence[int], mean_fd: Sequence[float]) -> Dict[str, Optional[float]]:
    """C2 descriptive: mean framewise displacement for ASD and TC participants."""
    y = labels_from_dx(dx_group)
    fd = np.asarray(mean_fd, dtype=float)
    out: Dict[str, Optional[float]] = {}
    for name, cls in (("asd", 1), ("tc", 0)):
        values = fd[(y == cls) & np.isfinite(fd)]
        out[name] = float(values.mean()) if values.size else None
    return out


def primary_analysis(
    dx_group: Sequence[int], scores: Sequence[float], sites: Sequence[str],
    ages: Sequence[float], **kwargs,
) -> Dict[str, object]:
    """§6 plus C3 and C4 on the participants that reached the model.

    Kill condition K5 is checked first: below ``MIN_PER_CLASS_OVERALL`` per
    class no verdict is issued. C4 can move the verdict from the pooled AUC to
    the within-site mean; ``endpoint`` records which one the verdict used.
    """
    y = labels_from_dx(dx_group)
    s = np.asarray(scores, dtype=float)
    site_arr = np.asarray(sites, dtype=str)
    age_arr = np.asarray(ages, dtype=float)
    n_asd, n_tc = _class_counts(y)

    result: Dict[str, object] = {"n_asd": n_asd, "n_tc": n_tc, "baseline_auc": BASELINE_AUC}
    if min(n_asd, n_tc) < MIN_PER_CLASS_OVERALL:
        result.update(verdict=UNDERPOWERED, endpoint=None)
        return result

    pooled = pooled_auc_with_ci(y, s, **kwargs)
    confound = site_confound(y, s, site_arr, pooled["auc"], **kwargs)
    if confound["flagged"]:
        endpoint = "within_site_mean"
        decided = verdict(confound["within_site_mean_auc"], confound["ci_lower"])
    else:
        endpoint = "pooled"
        decided = verdict(pooled["auc"], pooled["ci_lower"])

    result.update(
        verdict=decided,
        endpoint=endpoint,
        pooled=pooled,
        site_confound=confound,
        operating_point=operating_point(y, s),
        per_site=per_site_auc(y, s, site_arr),
        age_bands=age_bands(y, s, age_arr, **kwargs),
    )
    return result
