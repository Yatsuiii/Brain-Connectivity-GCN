"""
Tests for the preregistered analysis and the freeze guard.

The decision boundaries tested here are the preregistration's [PROPOSED]
values (§6, §7). If the frozen plan changes them, these tests change with it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brain_gcn.external_validation import analysis as A
from brain_gcn.external_validation import prereg

REPO = Path(__file__).resolve().parents[1]


def separable(n_per_class, shift, seed=0):
    """dx_group and scores where ASD scores sit ``shift`` above TC scores."""
    rng = np.random.default_rng(seed)
    dx = np.r_[np.ones(n_per_class, int), np.full(n_per_class, 2)]
    scores = np.r_[rng.normal(shift, 1, n_per_class), rng.normal(0, 1, n_per_class)]
    return dx, scores


class TestVerdict:
    """§6: exhaustive, mutually exclusive, boundaries exactly as written."""

    def test_replicated_at_the_noninferiority_boundary(self):
        assert A.verdict(0.6798, 0.61) == A.REPLICATED

    def test_just_below_the_boundary_is_attenuated(self):
        assert A.verdict(0.6797, 0.61) == A.ATTENUATED

    def test_ci_floor_is_strict(self):
        assert A.verdict(0.75, 0.60) == A.ATTENUATED

    def test_ci_touching_chance_fails_regardless_of_point_estimate(self):
        assert A.verdict(0.80, 0.50) == A.FAILED

    def test_edited_baseline_does_not_shift_boundary_by_float_error(self):
        assert A.verdict(0.68, 0.61, baseline=0.73, margin=0.05) == A.REPLICATED

    @pytest.mark.parametrize("auc", np.linspace(0.3, 1.0, 15))
    @pytest.mark.parametrize("ci_lower", np.linspace(0.3, 1.0, 15))
    def test_always_one_of_three(self, auc, ci_lower):
        assert A.verdict(auc, ci_lower) in {A.REPLICATED, A.ATTENUATED, A.FAILED}


class TestBootstrap:
    def test_ci_is_deterministic_and_brackets_the_estimate(self):
        dx, scores = separable(150, 0.8)
        y = A.labels_from_dx(dx)
        first = A.pooled_auc_with_ci(y, scores, n_resamples=500)
        second = A.pooled_auc_with_ci(y, scores, n_resamples=500)
        assert first == second
        assert first["ci_lower"] < first["auc"] < first["ci_upper"]

    def test_resamples_preserve_class_counts(self):
        y = np.r_[np.ones(30, int), np.zeros(70, int)]
        seen = []
        A.bootstrap_ci(y, y.astype(float), lambda idx: seen.append(y[idx].sum()) or 0.0,
                       strata=y, n_resamples=50)
        assert set(seen) == {30}

    def test_labels_reject_unknown_codes(self):
        with pytest.raises(ValueError):
            A.labels_from_dx([1, 2, 3])


class TestControls:
    def test_operating_point_uses_the_fixed_threshold(self):
        y = np.array([1, 1, 0, 0])
        op = A.operating_point(y, np.array([0.9, 0.4, 0.6, 0.1]))
        assert op == {"threshold": 0.5, "accuracy": 0.5, "sensitivity": 0.5, "specificity": 0.5}

    def test_small_sites_get_counts_only(self):
        dx, scores = separable(12, 1.0)
        y = A.labels_from_dx(dx)
        sites = np.array(["big"] * 10 + ["small"] * 2 + ["big"] * 10 + ["small"] * 2)
        rows = {r["site"]: r for r in A.per_site_auc(y, scores, sites)}
        assert rows["big"]["auc"] is not None
        assert rows["small"]["auc"] is None and rows["small"]["n_asd"] == 2

    def test_site_confound_flags_a_model_that_only_recognises_sites(self):
        # Site A is 80% ASD and scores high for everyone; site B is 20% ASD and
        # scores low for everyone. Within each site the score is uninformative.
        rng = np.random.default_rng(6)
        y = np.r_[np.ones(80), np.zeros(20), np.ones(20), np.zeros(80)].astype(int)
        sites = np.array(["A"] * 100 + ["B"] * 100)
        scores = np.where(sites == "A", 1.0, 0.0) + rng.normal(0, 0.05, 200)
        pooled = A.pooled_auc_with_ci(y, scores, n_resamples=200)["auc"]
        out = A.site_confound(y, scores, sites, pooled, n_resamples=200)
        assert pooled > 0.75
        assert abs(out["within_site_mean_auc"] - 0.5) < 0.15
        assert out["flagged"]

    def test_site_confound_quiet_when_signal_is_within_site(self):
        dx, scores = separable(100, 1.2, seed=7)
        y = A.labels_from_dx(dx)
        sites = np.tile(["A", "B"], 100)
        pooled = A.pooled_auc_with_ci(y, scores, n_resamples=200)["auc"]
        assert not A.site_confound(y, scores, sites, pooled, n_resamples=200)["flagged"]

    def test_underfilled_age_bands_report_counts_only(self):
        dx, scores = separable(40, 1.0)
        y = A.labels_from_dx(dx)
        ages = np.r_[np.full(30, 10.0), np.full(10, 40.0), np.full(30, 10.0), np.full(10, 40.0)]
        rows = A.age_bands(y, scores, ages, n_resamples=100)
        under_12, adults = rows[0], rows[3]
        assert under_12["auc"] is not None
        assert adults["auc"] is None and adults["n_asd"] == 10

    def test_selection_bias_detects_diagnosis_dependent_inclusion(self):
        included = np.r_[np.ones(100, bool), np.zeros(100, bool)]
        dx = np.r_[np.full(80, 1.0), np.full(20, 2.0), np.full(20, 1.0), np.full(80, 2.0)]
        nan = np.full(200, np.nan)
        out = A.selection_bias(included, dx, nan, nan, np.array([""] * 200))
        assert out["dx_by_inclusion_p"] < 1e-6
        assert out["conditional_on_preprocessing"]
        assert out["age_by_inclusion_p"] is None

    def test_selection_bias_quiet_when_balanced(self):
        included = np.tile([True, False], 100)
        dx = np.repeat([1.0, 2.0], 100)
        ages = np.linspace(6, 30, 200)
        sites = np.array(["S1", "S2"] * 100)
        out = A.selection_bias(included, dx, ages, np.ones(200), sites)
        assert out["dx_by_inclusion_p"] > 0.5
        assert not out["conditional_on_preprocessing"]

    def test_mean_fd_by_class_ignores_missing(self):
        out = A.mean_fd_by_class([1, 1, 2], [0.2, np.nan, 0.4])
        assert out == {"asd": 0.2, "tc": 0.4}


class TestPrimaryAnalysis:
    def test_underpowered_issues_no_verdict(self):
        dx, scores = separable(A.MIN_PER_CLASS_OVERALL - 1, 1.0)
        out = A.primary_analysis(dx, scores, ["S"] * len(dx), [10.0] * len(dx))
        assert out["verdict"] == A.UNDERPOWERED and out["endpoint"] is None

    def test_full_result_uses_pooled_endpoint_without_site_confound(self):
        dx, scores = separable(150, 1.0)
        sites = np.tile(["A", "B", "C"], 100)
        out = A.primary_analysis(dx, scores, sites, np.full(300, 14.0), n_resamples=200)
        assert out["endpoint"] == "pooled"
        assert out["verdict"] == A.REPLICATED
        assert {"pooled", "site_confound", "operating_point", "per_site",
                "age_bands"} <= set(out)

    def test_site_confound_moves_the_verdict_to_within_site(self):
        rng = np.random.default_rng(8)
        dx = np.r_[np.ones(120), np.full(30, 2), np.ones(30), np.full(120, 2)].astype(int)
        sites = np.array(["A"] * 150 + ["B"] * 150)
        scores = np.where(sites == "A", 1.0, 0.0) + rng.normal(0, 0.05, 300)
        out = A.primary_analysis(dx, scores, sites, np.full(300, 14.0), n_resamples=200)
        assert out["endpoint"] == "within_site_mean"
        assert out["verdict"] == A.FAILED


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                   cwd=repo, check=True, capture_output=True)


class TestFreezeGuard:
    """§0: evaluation refuses to run until the plan is frozen."""

    DOC = "| Field | Value |\n| --- | --- |\n| Frozen commit | `{sha}` |\n"

    def repo(self, tmp_path):
        git(tmp_path, "init", "-q")
        doc = tmp_path / prereg.PREREG_PATH
        doc.parent.mkdir(parents=True)
        doc.write_text(self.DOC.format(sha="UNFROZEN"))
        git(tmp_path, "add", ".")
        git(tmp_path, "commit", "-q", "-m", "plan")
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                             capture_output=True, text=True, check=True).stdout.strip()
        return doc, sha

    def test_parses_recorded_sha(self):
        assert prereg.frozen_sha(self.DOC.format(sha="UNFROZEN")) is None
        assert prereg.frozen_sha(self.DOC.format(sha="abc1234")) == "abc1234"

    def test_committed_plan_parses(self):
        text = (REPO / prereg.PREREG_PATH).read_text(encoding="utf-8")
        recorded = prereg.frozen_sha(text)
        assert recorded is None or len(recorded) >= prereg.MIN_SHA_LENGTH

    def test_unfrozen_plan_refuses(self, tmp_path):
        _, sha = self.repo(tmp_path)
        with pytest.raises(prereg.PreregistrationNotFrozen, match="not frozen"):
            prereg.require_frozen(sha, tmp_path)

    def test_frozen_plan_accepts_matching_short_sha(self, tmp_path):
        doc, sha = self.repo(tmp_path)
        doc.write_text(self.DOC.format(sha=sha))
        assert prereg.require_frozen(sha[:8], tmp_path) == sha

    def test_mismatched_or_missing_sha_refuses(self, tmp_path):
        doc, sha = self.repo(tmp_path)
        doc.write_text(self.DOC.format(sha=sha))
        with pytest.raises(prereg.PreregistrationNotFrozen, match="does not match"):
            prereg.require_frozen("0000000", tmp_path)
        with pytest.raises(prereg.PreregistrationNotFrozen, match="--prereg-sha"):
            prereg.require_frozen(None, tmp_path)

    def test_sha_not_in_history_refuses(self, tmp_path):
        doc, _ = self.repo(tmp_path)
        doc.write_text(self.DOC.format(sha="deadbeefdeadbeef"))
        with pytest.raises(prereg.PreregistrationNotFrozen, match="history"):
            prereg.require_frozen("deadbeef", tmp_path)


class TestEvalScript:
    """eval_abide2.py: refusal paths that must hold before and after the freeze."""

    @pytest.fixture
    def script(self):
        sys.path.insert(0, str(REPO))
        import eval_abide2
        return eval_abide2

    def test_evaluate_refuses_a_wrong_sha(self, script, tmp_path):
        assert script.main(["--data-dir", str(tmp_path), "evaluate",
                            "--prereg-sha", "0000000"]) == 3

    def test_completeness_counts_only_unresolved_eligible_participants(self, script):
        cohort = [{"participant_id": p, "exclusion": e}
                  for p, e in [("1", ""), ("2", ""), ("3", ""), ("4", ""), ("5", "rule 1: x")]]
        extraction = {
            "1": {"path": "a.1D", "exclusion": "", "error": ""},
            "2": {"path": "", "exclusion": "rule 4: short", "error": ""},
            "3": {"path": "", "exclusion": "", "error": "URLError: timeout"},
        }
        assert script._completeness(cohort, extraction) == ["3", "4"]
