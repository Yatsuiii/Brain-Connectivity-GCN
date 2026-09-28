"""
Tests for the ABIDE II loader (brain_gcn/external_validation/abide2.py).

All fixtures are synthetic: no ABIDE data is committed and no test touches the
network. The S3 layout mirrored here was verified against the public bucket on
2026-09-28 (see the module docstring).
"""

import io
import sys
import urllib.error
import urllib.parse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from brain_gcn.external_validation import abide2

REPO = Path(__file__).resolve().parents[1]
NS = "http://s3.amazonaws.com/doc/2006-03-01/"
FP = abide2.FMRIPREP_PREFIX


def list_xml(keys=(), prefixes=(), token=None):
    """A ListObjectsV2 page, namespaced like the real S3 responses."""
    body = "".join(f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in keys)
    body += "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes)
    if token:
        body += f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>"
    return f'<?xml version="1.0"?><ListBucketResult xmlns="{NS}">{body}</ListBucketResult>'.encode()


def bold_key(sub, stem):
    return f"{FP}sub-{sub}/{stem}{abide2.BOLD_SUFFIX}"


def conf_key(sub, stem, suffix="_desc-confounds_regressors.tsv"):
    return f"{FP}sub-{sub}/{stem}{suffix}"


class TestListing:
    """S3 ListObjectsV2 parsing and pagination."""

    def test_parses_keys_sizes_prefixes_and_token(self):
        xml = list_xml(keys=[("a/b.tsv", 12)], prefixes=["a/site/"], token="tok")
        objects, prefixes, token = abide2.parse_list_response(xml)
        assert objects == [("a/b.tsv", 12)]
        assert prefixes == ["a/site/"]
        assert token == "tok"

    def test_last_page_has_no_token(self):
        _, _, token = abide2.parse_list_response(list_xml(keys=[("k", 1)]))
        assert token is None

    def test_follows_continuation_tokens(self):
        pages = {None: list_xml(keys=[("k1", 1)], token="t2"),
                 "t2": list_xml(keys=[("k2", 2)])}

        def fetch(url):
            return pages["t2" if "continuation-token=t2" in url else None]

        objects, _ = abide2.list_objects("p/", fetch)
        assert objects == [("k1", 1), ("k2", 2)]


class TestRunSelection:
    """§4.1 rules 2-3: one rest run per participant, BOLD and confounds paired."""

    def test_picks_lowest_session_then_run_among_complete_runs(self):
        s11 = "ses-1/func/sub-1_ses-1_task-rest_run-1"
        s12 = "ses-1/func/sub-1_ses-1_task-rest_run-2"
        s21 = "ses-2/func/sub-1_ses-2_task-rest_run-1"
        objects = [
            (bold_key("1", s11), 100),                       # no confounds -> ineligible
            (bold_key("1", s12), 200), (conf_key("1", s12), 1),
            (bold_key("1", s21), 300), (conf_key("1", s21), 1),
        ]
        run = abide2.select_runs(objects)["1"]
        assert (run.session, run.run) == ("1", "2")
        assert run.bold_bytes == 200

    def test_run_numbers_sort_naturally(self):
        s2 = "func/sub-7_task-rest_run-2"
        s10 = "func/sub-7_task-rest_run-10"
        objects = [(bold_key("7", s10), 1), (conf_key("7", s10), 1),
                   (bold_key("7", s2), 1), (conf_key("7", s2), 1)]
        assert abide2.select_runs(objects)["7"].run == "2"

    def test_accepts_both_confounds_names_and_missing_entities(self):
        stem = "func/sub-2_task-rest"
        objects = [(bold_key("2", stem), 5),
                   (conf_key("2", stem, "_desc-confounds_timeseries.tsv"), 1)]
        run = abide2.select_runs(objects)["2"]
        assert (run.session, run.run) == ("", "")

    def test_ignores_bold_without_confounds_and_non_rest_tasks(self):
        rest = "func/sub-3_task-rest"
        other = "func/sub-4_task-movie"
        objects = [(bold_key("3", rest), 1),
                   (f"{FP}sub-4/{other}{abide2.BOLD_SUFFIX}", 1),
                   (f"{FP}sub-4/{other}_desc-confounds_regressors.tsv", 1)]
        assert abide2.select_runs(objects) == {}

    def test_processed_participants_counts_any_output(self):
        objects = [(f"{FP}sub-9/anat/sub-9_desc-preproc_T1w.nii.gz", 1)]
        assert abide2.processed_participants(objects) == {"9"}


class TestParticipants:
    """§4.1 rule 1: labels from per-site BIDS participants.tsv."""

    TSV = (
        "site_id\tparticipant_id\tdx_group\tage_at_scan \tsex\n"
        "ABIDEII-X_1\t29006\t1\t48.0\t1\n"
        "ABIDEII-X_1\tsub-29007\t2\tn/a\t2\n"
        "ABIDEII-X_1\t29008\t3\t10.5\t1\n"
    ).encode()

    def test_strips_header_whitespace_and_id_prefix(self):
        rows = abide2.parse_participants(self.TSV, "ABIDEII-X_1")
        assert [r.participant_id for r in rows] == ["29006", "29007", "29008"]
        assert rows[0].age == 48.0  # read through the 'age_at_scan ' header

    def test_missing_and_invalid_values_become_none(self):
        rows = abide2.parse_participants(self.TSV, "ABIDEII-X_1")
        assert rows[1].age is None
        assert rows[2].dx_group is None  # 3 is not a valid dx_group
        assert [r.dx_group for r in rows[:2]] == [1, 2]

    def test_site_falls_back_to_folder_name(self):
        tsv = b"participant_id\tdx_group\n1\t1\n"
        (row,) = abide2.parse_participants(tsv, "ABIDEII-Y_1")
        assert row.site_id == "ABIDEII-Y_1"

    def test_latin1_byte_does_not_break_parsing(self):
        # ABIDEII-GU_1 holds one Latin-1 NBSP, in a free-text medication field.
        tsv = ("participant_id\tdx_group\tcurrent_medication_name\n"
               "7\t1\tdiphenhydramine HCl\xa0\n").encode("latin-1")
        (row,) = abide2.parse_participants(tsv, "ABIDEII-GU_1")
        assert (row.participant_id, row.dx_group) == ("7", 1)

    def test_repetition_time(self):
        assert abide2.parse_repetition_time(b'{"RepetitionTime": 2.0}') == 2.0
        assert abide2.parse_repetition_time(b'{"RepetitionTime": 0}') is None
        assert abide2.parse_repetition_time(b"not json") is None


class TestCohort:
    """Every participant is kept with the first rule they fail (needed for C2)."""

    def test_exclusion_reasons(self):
        people = [
            abide2.Participant("1", "S", 1, 10.0, 1),   # included
            abide2.Participant("2", "S", 2, 11.0, 1),   # never processed
            abide2.Participant("3", "S", 1, 12.0, 2),   # processed, no usable run
            abide2.Participant("4", "S", None, 9.0, 1), # invalid label
        ]
        run = abide2.RestRun("1", "", "", "s", "b", "c", 1)
        cohort = {e.participant.participant_id: e
                  for e in abide2.build_cohort(people, {"1": run}, {"1", "3", "5"})}
        assert cohort["1"].exclusion is None
        assert cohort["2"].exclusion == abide2.EXCLUDE_NOT_PROCESSED
        assert cohort["3"].exclusion == abide2.EXCLUDE_NO_BOLD
        assert cohort["4"].exclusion == abide2.EXCLUDE_DX
        assert cohort["5"].exclusion == abide2.EXCLUDE_NOT_LISTED

    def test_discover_end_to_end_with_fake_bucket(self):
        stem = "func/sub-1_task-rest"
        fmriprep = list_xml(keys=[(bold_key("1", stem), 7), (conf_key("1", stem), 1)])
        rawdata = list_xml(prefixes=[f"{abide2.RAWDATA_PREFIX}ABIDEII-A_1/",
                                     f"{abide2.RAWDATA_PREFIX}ABIDEII-B_1/"])
        files = {
            "ABIDEII-A_1/participants.tsv": b"participant_id\tdx_group\n1\t1\n",
            "ABIDEII-A_1/task-rest_bold.json": b'{"RepetitionTime": 2.0}',
            "ABIDEII-B_1/participants.tsv": b"participant_id\tdx_group\n2\t2\n",
        }

        def fetch(url):
            if "list-type=2" in url:
                return rawdata if "delimiter" in url else fmriprep
            for suffix, body in files.items():
                if url.endswith(suffix):
                    return body
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        cohort, site_tr = abide2.discover(fetch)
        by_id = {e.participant.participant_id: e for e in cohort}
        assert by_id["1"].exclusion is None and by_id["1"].run.bold_bytes == 7
        assert by_id["2"].exclusion == abide2.EXCLUDE_NOT_PROCESSED
        assert site_tr == {"ABIDEII-A_1": 2.0, "ABIDEII-B_1": None}


    def test_site_without_labels_is_excluded_by_site_not_dropped(self):
        # ABIDEII-ETHZ_1, -UCLA_Long and -UPSM_Long have imaging but no
        # participants.tsv; 'sidecards' is not a site at all.
        raw = abide2.RAWDATA_PREFIX
        fmriprep = list_xml(keys=[(f"{FP}sub-5/anat/sub-5_T1w.nii.gz", 1)])
        listings = {
            raw: list_xml(prefixes=[f"{raw}ABIDEII-Z_1/", f"{raw}sidecards/"]),
            f"{raw}ABIDEII-Z_1/": list_xml(prefixes=[f"{raw}ABIDEII-Z_1/sub-5/",
                                                     f"{raw}ABIDEII-Z_1/sub-6/"]),
        }

        def fetch(url):
            if "list-type=2" in url:
                if "delimiter" not in url:
                    return fmriprep
                prefix = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["prefix"][0]
                return listings[prefix]
            if url.endswith("task-rest_bold.json"):
                return b'{"RepetitionTime": 2.5}'
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        cohort, site_tr = abide2.discover(fetch)
        by_id = {e.participant.participant_id: e for e in cohort}
        assert set(by_id) == {"5", "6"}
        assert all(e.exclusion == abide2.EXCLUDE_NO_SITE_LABELS for e in cohort)
        assert by_id["5"].participant.site_id == "ABIDEII-Z_1"
        assert by_id["5"].processed and not by_id["6"].processed
        assert site_tr == {"ABIDEII-Z_1": 2.5}


class TestSignalProcessing:
    """§4.2 nuisance model and band-pass."""

    def confounds(self, n=60, drop=()):
        rng = np.random.default_rng(0)
        cols = abide2.MOTION_COLUMNS + abide2.ACOMPCOR_COLUMNS + (abide2.GLOBAL_SIGNAL_COLUMN,)
        frame = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
        return frame.drop(columns=list(drop))

    def test_friston24_layout(self):
        motion = np.arange(12, dtype=float).reshape(4, 3)
        out = abide2.friston24(motion)
        assert out.shape == (4, 12)
        assert np.all(out[0, 3:6] == 0)                     # first derivative row is 0
        np.testing.assert_allclose(out[1:, 3:6], np.diff(motion, axis=0))
        np.testing.assert_allclose(out[:, 6:9], motion ** 2)
        np.testing.assert_allclose(out[:, 9:12], out[:, 3:6] ** 2)

    def test_design_has_33_columns_starting_with_intercept(self):
        design = abide2.nuisance_design(self.confounds())
        assert design.shape == (60, 3 + 24 + abide2.N_ACOMPCOR + 1)
        assert np.all(design[:, 0] == 1)

    def test_missing_regressor_is_named(self):
        with pytest.raises(ValueError, match="a_comp_cor_04"):
            abide2.nuisance_design(self.confounds(drop=("a_comp_cor_04",)))

    def test_nan_first_row_is_zeroed(self):
        frame = self.confounds()
        frame.loc[0, "trans_x"] = np.nan
        assert np.isfinite(abide2.nuisance_design(frame)).all()

    def test_regression_removes_exactly_the_design_space(self):
        rng = np.random.default_rng(1)
        design = abide2.nuisance_design(self.confounds())
        noise = rng.normal(size=(60, 5))
        q, _ = np.linalg.qr(design)
        orthogonal = noise - q @ (q.T @ noise)
        signals = design @ rng.normal(size=(design.shape[1], 5)) + orthogonal
        np.testing.assert_allclose(abide2.regress_out(signals, design), orthogonal, atol=1e-8)

    def test_bandpass_keeps_in_band_and_removes_the_rest(self):
        n, tr = 256, 2.0                        # power of two: no padding, exact bins
        t = np.arange(n) * tr
        f_in, f_high, f_low = 25 / (n * tr), 100 / (n * tr), 2 / (n * tr)
        assert 0.01 <= f_in <= 0.1 and f_high > 0.1 and f_low < 0.01
        keep = np.sin(2 * np.pi * f_in * t)
        signal = keep + np.sin(2 * np.pi * f_high * t) + np.sin(2 * np.pi * f_low * t) + 5.0
        out = abide2.ideal_bandpass(signal[:, None], tr, *abide2.BANDPASS_HZ)[:, 0]
        np.testing.assert_allclose(out, keep, atol=1e-8)

    def test_bandpass_preserves_length_when_padding(self):
        out = abide2.ideal_bandpass(np.random.default_rng(2).normal(size=(152, 3)), 2.0, 0.01, 0.1)
        assert out.shape == (152, 3)


def nifti(data, zooms=(2.0, 2.0, 2.0, 2.0), time_unit="sec", origin=(0.0, 0.0, 0.0)):
    affine = np.diag([*zooms[:3], 1.0])
    affine[:3, 3] = origin
    img = nib.Nifti1Image(np.asarray(data), affine)
    if np.asarray(data).ndim == 4:
        img.header.set_zooms(zooms)
    img.header.set_xyzt_units("mm", time_unit)
    return img


class TestRoiMeans:
    """Column i must hold label i + 1: the ABIDE I '#1 ... #200' order."""

    def test_exact_means_in_label_order_with_empty_label(self):
        labels = np.zeros((4, 4, 4), dtype=np.int16)
        labels[:2] = 3            # label 3 first in memory, to catch sorting bugs
        labels[2:, :2] = 1
        # label 2 has no voxels
        signal = {1: np.arange(6.0), 3: np.arange(6.0) * -2}
        bold = np.zeros((4, 4, 4, 6))
        for lab, ts in signal.items():
            bold[labels == lab] = ts
        out = abide2.roi_means(nifti(bold), nifti(labels), n_rois=3)
        assert out.shape == (6, 3)
        np.testing.assert_allclose(out[:, 0], signal[1])
        np.testing.assert_allclose(out[:, 1], 0.0)         # empty label -> zero column
        np.testing.assert_allclose(out[:, 2], signal[3])

    def test_coarser_atlas_is_resampled_to_the_bold_grid(self):
        # A 4 mm atlas offset by 1 mm maps each atlas voxel onto a 2x2x2 block
        # of 2 mm BOLD voxels.
        atlas = np.arange(1, 9, dtype=np.int16).reshape(2, 2, 2)
        atlas_img = nifti(atlas, zooms=(4.0, 4.0, 4.0), origin=(1.0, 1.0, 1.0))
        rng = np.random.default_rng(3)
        series = rng.normal(size=(8, 5))
        bold = np.zeros((4, 4, 4, 5))
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    bold[x, y, z] = series[atlas[x // 2, y // 2, z // 2] - 1]
        out = abide2.roi_means(nifti(bold), atlas_img, n_rois=8)
        np.testing.assert_allclose(out, series.T, atol=1e-6)

    def test_header_tr_units(self):
        assert abide2.header_tr(nifti(np.zeros((2, 2, 2, 3)))) == 2.0
        img = nifti(np.zeros((2, 2, 2, 3)), zooms=(2.0, 2.0, 2.0, 2000.0), time_unit="msec")
        assert abide2.header_tr(img) == 2.0


class TestWrite1D:
    """Output must load exactly the way predict._load_1d loads ABIDE I files."""

    def test_round_trip_through_default_loadtxt(self, tmp_path):
        data = np.random.default_rng(4).normal(size=(60, 200)) * 10
        path = abide2.write_1d(data, tmp_path / "x_rois_cc200.1D")
        first = path.read_text().splitlines()[0]
        assert first.startswith("#1\t#2\t") and first.endswith("#200")
        loaded = np.loadtxt(path, dtype=np.float32)
        assert loaded.shape == (60, 200)
        np.testing.assert_allclose(loaded, data, atol=1e-5)


class TestExtractParticipant:
    """§4.1 rules 2 and 4-6 plus the full §4.2 path, on synthetic files."""

    N_VOL = 60

    @pytest.fixture
    def atlas(self):
        labels = (np.arange(10 * 10 * 8) // 4 + 1).reshape(10, 10, 8).astype(np.int16)
        return nifti(labels)

    def serve(self, tmp_path, n_vol=N_VOL, tr=2.0, conf_rows=None, drop=()):
        rng = np.random.default_rng(5)
        bold_img = nifti(rng.normal(size=(10, 10, 8, n_vol)).astype(np.float32),
                         zooms=(2.0, 2.0, 2.0, tr))
        bold_file = tmp_path / "src_bold.nii.gz"
        nib.save(bold_img, str(bold_file))
        frame = TestSignalProcessing().confounds(n=conf_rows or n_vol, drop=drop)
        frame["framewise_displacement"] = np.r_[np.nan, np.full(len(frame) - 1, 0.2)]
        buf = io.StringIO()
        frame.to_csv(buf, sep="\t", index=False, na_rep="n/a")
        run = abide2.RestRun("42", "", "", "sub-42_task-rest",
                             bold_key("42", "func/sub-42_task-rest"),
                             conf_key("42", "func/sub-42_task-rest"), 1)
        calls = []

        def fetch(url):
            calls.append(url)
            if url.endswith(abide2.BOLD_SUFFIX):
                return bold_file.read_bytes()
            return buf.getvalue().encode()

        return run, fetch, calls

    def test_writes_a_loadable_200_column_file(self, tmp_path, atlas):
        run, fetch, _ = self.serve(tmp_path)
        work = tmp_path / "work"
        out = abide2.extract_participant(run, 2.0, atlas, tmp_path / "out", work, fetch)
        assert out.exclusion is None
        loaded = np.loadtxt(out.path, dtype=np.float32)
        assert loaded.shape == (self.N_VOL, 200)
        assert out.n_empty_rois == 0
        assert out.mean_fd == pytest.approx(0.2)
        assert not any(p.name.endswith(abide2.BOLD_SUFFIX) for p in work.iterdir())

    def test_short_series_is_rule_4(self, tmp_path, atlas):
        run, fetch, _ = self.serve(tmp_path, n_vol=abide2.MIN_VOLUMES - 1)
        out = abide2.extract_participant(run, 2.0, atlas, tmp_path / "o", tmp_path / "w", fetch)
        assert out.exclusion == abide2.EXCLUDE_SHORT and out.path is None

    def test_tr_disagreement_is_rule_5(self, tmp_path, atlas):
        run, fetch, _ = self.serve(tmp_path, tr=2.0)
        out = abide2.extract_participant(run, 2.5, atlas, tmp_path / "o", tmp_path / "w", fetch)
        assert out.exclusion == abide2.EXCLUDE_TR

    def test_unknown_site_tr_is_accepted_and_recorded(self, tmp_path, atlas):
        run, fetch, _ = self.serve(tmp_path)
        out = abide2.extract_participant(run, None, atlas, tmp_path / "o", tmp_path / "w", fetch)
        assert out.exclusion is None and out.site_tr is None and out.tr == 2.0

    def test_row_mismatch_is_rule_6(self, tmp_path, atlas):
        run, fetch, _ = self.serve(tmp_path, conf_rows=self.N_VOL - 1)
        out = abide2.extract_participant(run, 2.0, atlas, tmp_path / "o", tmp_path / "w", fetch)
        assert out.exclusion == abide2.EXCLUDE_ROWS

    def test_missing_regressors_exclude_before_the_bold_download(self, tmp_path, atlas):
        run, fetch, calls = self.serve(tmp_path, drop=("global_signal",))
        out = abide2.extract_participant(run, 2.0, atlas, tmp_path / "o", tmp_path / "w", fetch)
        assert out.exclusion == abide2.EXCLUDE_REGRESSORS
        assert not any(u.endswith(abide2.BOLD_SUFFIX) for u in calls)


def test_min_volumes_matches_predict():
    """The loader's floor must equal the one predict._load_1d enforces."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO))
    import predict

    assert abide2.MIN_VOLUMES == predict._WINDOW_LEN + predict._STEP
    assert abide2.N_ROIS == 200
