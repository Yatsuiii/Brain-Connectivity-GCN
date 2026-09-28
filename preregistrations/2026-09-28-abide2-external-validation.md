# Preregistration: zero-shot external validation on ABIDE II

| Field | Value |
| --- | --- |
| Status | **DRAFT — not frozen.** No ABIDE II subject may be scored until frozen. |
| Drafted | 2026-09-28 |
| Model under test | 20-model CC200 leave-one-site-out ensemble, `hf_space/checkpoints/cc200_*.ckpt`, as of commit `2520b4d` |
| Frozen commit | `UNFROZEN` |
| Frozen on | — |

## 0. Freeze procedure

1. Review every value marked **[PROPOSED]** and edit it if needed.
2. Resolve the pre-freeze checks in §2.3.
3. Commit the plan with the message `Freeze ABIDE II preregistration`.
4. In a follow-up commit, replace `UNFROZEN` above with that commit's SHA and fill in the date.
5. From then on the plan is append-only: any change goes in §11 with a date and a
   reason, and results are still reported against the frozen plan.

`eval_abide2.py --evaluate` refuses to score anything unless `--prereg-sha` matches the
frozen commit recorded above. The guard prevents accidental analysis before the
freeze. Git history is the audit trail for everything else.

## 1. Question

Does the ABIDE I–trained CC200 ensemble separate autistic participants from typical
controls on ABIDE II, a set of sites and participants it has never seen, without
retraining, recalibration, harmonization, or threshold tuning?

Secondary (Arm B): does parcellating each participant with a whole-brain
segmentation model, instead of the CC200 template, improve cross-site
generalization when preprocessing is held fixed?

## 2. What is known at registration

### 2.1 The model

- Four-site leave-one-site-out evaluation (NYU, USM, UCLA, UM): AUC 0.7872, N=529.
- 20-site leave-one-site-out experiment: AUC 0.7298, N=1,102.
- Each checkpoint was trained with one ABIDE I site held out. None has seen ABIDE II.

### 2.2 The ABIDE II data

Established from listings of the public `fcp-indi` bucket on 2026-09-28. No imaging
data was analyzed and no participant was scored.

- **There are no C-PAC ROI time series for ABIDE II.** `ABIDE2/Outputs/cpac/` is
  empty. The previous `eval_abide2.py` requested that path and a phenotype file at
  `ABIDE2/RawData/Phenotypic_V1_0b.csv` that returns HTTP 404, so it never produced
  a result.
- fMRIPrep outputs exist for 732 participants. 586 have
  `space-MNI152NLin2009cAsym_desc-preproc_bold`, 584 have a confounds file, and 150
  have crash logs.
- Labels come from per-site BIDS `participants.tsv` files, column `dx_group`, coded
  1 = ASD and 2 = typical control, the same coding as ABIDE I. 16 of the 19 site
  folders have one. **ABIDEII-ETHZ_1, ABIDEII-UCLA_Long, and ABIDEII-UPSM_Long (75
  participants, 56 of them processed by fMRIPrep) have imaging but no labels
  anywhere in the public bucket.** See P4.
- This is an older fMRIPrep release. The confounds file is named
  `desc-confounds_regressors.tsv`, the Friston-24 motion expansion is not
  precomputed, and volumetric BOLD has no JSON sidecar.
- Cohort after the §4.1 rules, from `eval_abide2.py discover` (listings and
  phenotype files only; no BOLD read, no model loaded): see §2.4.
- **Pipeline QA.** One participant was processed through §4.2 to check the loader
  on real data:
  - 152 volumes; the header TR matched the site's BIDS TR (2.0 s); 0 empty ROIs.
  - `predict._load_1d` accepted the file, and its header is byte-identical to the
    ABIDE I files.
  - Mean off-diagonal FC was r = +0.014 and mean |r| = 0.189, against +0.004 and
    0.179 for an ABIDE I C-PAC file.
  - 99.2% of power fell in 0.01–0.1 Hz, against 99.6%.

  This is one file on each side, a sanity check and not a comparison. No model
  was loaded and no participant was scored.

### 2.3 Pre-freeze checks

- **P1 — baseline computation.** Confirm that 0.7298 is a single AUC over the
  concatenated held-out predictions. If it is instead the mean of per-site AUCs, the
  primary endpoint in §6 switches to the within-site mean so the comparison stays
  like-for-like.
- **P2 — Arm B model eligibility.** See §8.1.
- **P3 — proposed values.** Review every **[PROPOSED]** item.
- **P4 — unlabeled sites.** Choose one:
  - (a) keep ETHZ_1, UCLA_Long, and UPSM_Long excluded under rule 1, as §4.1
    currently specifies; or
  - (b) before freezing, obtain their labels from the ABIDE II composite
    phenotype file (distributed through NITRC, subject to the ABIDE data-use
    terms) and add that file as a documented label source.

  Option (b) adds up to 52 participants with usable runs. It cannot be taken
  after the freeze.

### 2.4 Cohort known at registration

From `eval_abide2.py discover` on 2026-09-28. Counts only; no outcome data.

| | N |
| --- | ---: |
| Participants across all site folders | 1,082 |
| Excluded, rule 1: site has no public labels (P4) | 75 |
| Excluded, rule 2: no fMRIPrep output | 331 |
| Excluded, rule 2: no MNI preprocessed BOLD with confounds | 144 |
| **Eligible for extraction** | **532 (ASD 243, TC 289)** |

Rules 4–6 are applied during extraction and may remove a few more. Streaming the
eligible cohort's BOLD is about 167 GB.

Eligible participants by site:

| Site | ASD | TC | Site | ASD | TC |
| --- | ---: | ---: | --- | ---: | ---: |
| KKI_1 | 52 | 141 | NYU_1 | 14 | 8 |
| GU_1 | 28 | 30 | UCD_1 | 16 | 9 |
| IP_1 | 20 | 27 | BNI_1 | 9 | 4 |
| EMC_1 | 21 | 19 | UCLA_1 | 8 | 6 |
| TCD_1 | 21 | 21 | **KUL_3** | **24** | **0** |
| OHSU_1 | 15 | 23 | **NYU_2** | **15** | **0** |
| | | | USM_1 | 0 | 1 |

Eligible age: median 10.8 years, range 5.2–64.0. By band: 339 under 12, 114 aged
12–17, 60 aged 18–29, and 19 aged 30 or over.

## 3. Hypotheses

- **H1 (Arm A, primary).** Pooled ROC AUC on ABIDE II is non-inferior to the fixed
  baseline: AUC ≥ 0.6798, i.e. 0.7298 minus a margin of 0.05 **[PROPOSED]**.
- **H2 (Arm B).** With identical data, folds, and training, subject-derived
  parcellation yields a higher leave-one-site-out AUC than CC200.

## 4. Arm A — zero-shot external validation

### 4.1 Inclusion

Rules are applied in order. The first rule a participant fails is recorded as the
reason for exclusion.

1. Listed in a site `participants.tsv` with `dx_group` of 1 or 2. Participants at
   sites with no public `participants.tsv` fail this rule (§2.3 P4).
2. Has a `space-MNI152NLin2009cAsym_desc-preproc_bold` run with a confounds file
   for the same run that contains every §4.2 regressor.
3. One run per participant: the lowest session, then the lowest run, among runs
   that meet rule 2.
4. At least 53 volumes. This is the model's minimum: a 50-volume window plus a
   3-volume step.
5. The BOLD header has a TR (band-pass filtering needs one), and it is within 1% of
   the site's raw BIDS `RepetitionTime`. If the site JSON is missing, the
   participant is included and the TR recorded as unverified.
6. The confounds file has one row per BOLD volume.

There is no exclusion on motion, image quality, or model output. Motion is reported
instead (§7, C2).

### 4.2 ROI time series

The goal is to approximate the ABIDE I C-PAC `filt_global` inputs the model was
trained on.

- **Atlas.** The CC200 file ABIDE I used
  (`ABIDE_Initiative/Resources/cc200_roi_atlas.nii.gz`: 200 contiguous labels, 3 mm),
  resampled to the BOLD grid by nearest neighbour.
- **Signal.** The mean over every atlas voxel of each label. Columns are ordered by
  label, 1 to 200, matching the `#1 … #200` header of the ABIDE I `.1D` files. A
  label with no voxels becomes a zero column and is kept; the model's per-ROI
  z-scoring already handles zero-variance columns.
- **Nuisance regression.** One ordinary least squares fit per participant, with
  these regressors:
  - intercept, linear trend, and quadratic trend;
  - Friston-24 motion: the six rigid-body parameters, their backward differences
    (first row 0), and the squares of both;
  - aCompCor components 00 to 04;
  - global signal.
- **Band-pass.** An ideal FFT filter keeping 0.01–0.1 Hz, applied after regression,
  which is C-PAC's order.
- No smoothing, scrubbing, or ICA-AROMA.
- **Output.** A tab-separated `.1D` file with 6 decimals and a `#1 … #200` header,
  format-compatible with the ABIDE I files and read by the unmodified `predict.py`
  path.

Known differences from C-PAC form the pipeline half of the joint shift tested by C1:

- fMRIPrep's registration instead of C-PAC's;
- the MNI152NLin2009cAsym 2 mm grid instead of the FSL MNI152 3 mm grid;
- fMRIPrep's aCompCor mask instead of C-PAC's;
- this implementation of regression and filtering instead of C-PAC's.

### 4.3 Inference

- The unmodified `predict_file` with all 20 checkpoints, averaging P(ASD).
- An operating threshold of 0.5, as in `predict.py`. It is reported, never tuned.
- No retraining, recalibration, harmonization (for example ComBat), or test-time
  adaptation on ABIDE II.

## 5. Fixed baseline

**0.7298**, from the 20-site ABIDE I leave-one-site-out experiment (N=1,102).

The four-site 0.7872 is not the comparison. It is a favorable subset of sites,
`CONTRIBUTING.md` requires the two results to be kept distinct, and 0.7298 is the
estimate over the model's whole training distribution.

## 6. Primary endpoint and decision rule

The endpoint is pooled ROC AUC over all included participants. Its 95% CI comes from
2,000 bootstrap resamples, stratified by class, with seed 20260928.

The verdicts are exhaustive and mutually exclusive **[PROPOSED]**:

| Verdict | Rule |
| --- | --- |
| **REPLICATED** | AUC ≥ 0.6798 and CI lower bound > 0.60 |
| **ATTENUATED** | CI lower bound > 0.50, but REPLICATED is not met |
| **FAILED** | CI lower bound ≤ 0.50 |

These are always reported: accuracy, sensitivity, and specificity at the 0.5
threshold, and per-site AUC.

## 7. Controls

### C1 — Site shift and pipeline shift at once

ABIDE II differs from ABIDE I both in sites and population and in preprocessing
(fMRIPrep instead of C-PAC). An ABIDE II drop alone cannot say which caused it.

The control uses ABIDE I itself. Each ABIDE I participant is scored only by the
checkpoint that held out their site, so every prediction is held out. This is done
once per ABIDE Preprocessed pipeline — C-PAC, CCS, DPARSF, and NIAK — using their
CC200 `filt_global` time series. Only participants present in all four pipelines are
used. The held-out AUC is reported for each pipeline.

- **Sanity check.** The C-PAC arm must reproduce the 0.7298 baseline within ±0.02
  **[PROPOSED]**. If it does not, the inference path differs from the one that
  produced the baseline, and kill condition K2 applies.
- **Interpretation [PROPOSED].** If the largest drop from C-PAC to another pipeline
  on ABIDE I is at least the ABIDE II drop (0.7298 minus the Arm A AUC), the ABIDE II
  result is not attributed to site or population shift.

### C2 — Selection bias from preprocessing failures

146 of the 732 processed participants have no BOLD output. If failure depends on
diagnosis, for example through more head motion in autistic children, the
evaluable sample is biased.

- Two comparisons: included vs. processed-but-excluded, and included vs. all
  labeled participants. Each compares diagnosis (χ²), age (Mann–Whitney U), sex
  (χ²), and site composition.
- **Rule [PROPOSED].** If the diagnosis-by-inclusion χ² gives p < 0.05, the primary
  result is labeled "conditional on successful preprocessing", and this comparison
  appears next to the AUC rather than in an appendix.
- Mean framewise displacement is reported by class among included participants. It
  is descriptive only.

### C3 — Age shift

The shift runs both ways. The eligible ABIDE II cohort is younger than ABIDE I
overall (median 10.8 years, with 339 of the 532 under 12), and it also has an adult
tail (79 participants aged 18 or over, up to 64).

- AUC with a 95% CI is reported in fixed `age_at_scan` bands: under 12, 12 to under
  18, 18 to under 30, and 30 or over.
- A band with fewer than 20 participants in either class is reported as counts
  only, with no AUC.
- Generalization is not claimed for any band whose CI lower bound is ≤ 0.5.

### C4 — Site confound

Pooled AUC can be inflated if the ensemble's scores shift by site and ASD prevalence
also differs by site. The model would then be rewarded for recognizing the site.

This risk is concrete in this cohort (§2.4):

- KKI_1 contributes 193 of the 532 participants, 73% of them controls.
- KUL_3 (24 ASD) and NYU_2 (15 ASD) contribute no controls at all.
- Only six sites meet the 10-per-class minimum: KKI_1, GU_1, IP_1, EMC_1, TCD_1,
  and OHSU_1, covering 418 participants.

- Per-site AUC is reported for sites with at least 10 participants per class,
  together with the sample-size-weighted mean of those within-site AUCs. Its CI
  comes from 2,000 bootstrap resamples stratified by site and class.
- **Rule [PROPOSED].** If pooled AUC exceeds the within-site mean by more than 0.05,
  the §6 verdict is computed on the within-site mean, and the discrepancy is
  reported.

## 8. Arm B — subject-derived parcellation (NVIDIA/MONAI model)

### 8.1 Model and eligibility

The model is MONAI's `wholeBrainSeg_Large_UNEST_segmentation`: a UNesT transformer
that segments 133 structures from T1-weighted MRI. It was trained on OASIS and CANDI
scans registered to MNI305.

**Pre-freeze check P2.** The bundle is published by the MONAI organization, which
NVIDIA co-founded with King's College London, not under NVIDIA's own namespace.
Before freezing, confirm that it satisfies the Nebius × NVIDIA hackathon's "NVIDIA
open-source model" requirement and that its license permits this use. If it does
not, name a replacement NVIDIA-published brain-MRI segmentation model here, or
remove Arm B.

### 8.2 Parcellation

- **Input.** Each included participant's
  `space-MNI152NLin2009cAsym_desc-preproc_T1w`. It is already in MNI space, as the
  bundle expects. The MNI305 vs. MNI152NLin2009cAsym difference is recorded as a
  limitation.
- **Grey matter only.** Drop every label whose name contains ventricle, white
  matter, CSF, vessel, optic chiasm, or background. The list of kept labels is
  written to `preregistrations/arm_b_labels.json` before any time series is
  extracted.
- The segmentation is resampled to the BOLD grid by nearest neighbour. ROI time
  series are cleaned exactly as in §4.2.

### 8.3 Design

The CC200 ensemble cannot take a different number of ROIs, so Arm B trains new
models. Both conditions share everything except the parcellation:

- the architecture and hyperparameters of the ABIDE I CC200 configuration;
- the same participants;
- the same leave-one-site-out folds, one per ABIDE II site with at least 10
  participants per class (participants from smaller sites are always in training);
- three seeds: 20260928, 20260929, and 20260930.

The two conditions are (i) CC200 and (ii) UNesT.

### 8.4 Endpoint

The endpoint is the paired difference in pooled leave-one-site-out AUC, UNesT minus
CC200, averaged across seeds. Its 95% CI comes from a paired bootstrap over
participants (2,000 resamples).

| Verdict | Rule |
| --- | --- |
| **SUPPORTED** | CI lower bound > 0 |
| **NULL** | CI contains 0 |
| **HARMFUL** | CI upper bound < 0 |

All three outcomes are reported.

### 8.5 Order

Arm B starts only after the Arm A verdict has been written to §10 and pushed. Arm B
results cannot be used to revise Arm A.

## 9. Kill conditions

- **K1 — Arm A FAILED.** Withdraw the external-generalization claim from the
  README and the model card. Publish the negative result with the full set of
  controls. Do not retrain, recalibrate, harmonize, or search thresholds on ABIDE II
  to rescue it.
- **K2 — C1 sanity check fails.** Stop. No ABIDE II number can be interpreted until
  the inference path reproduces the baseline.
- **K3 — Arm B NULL or HARMFUL.** Report it as such. Make no claim that
  subject-derived parcellation helps.
- **K4 — P2 unresolved by 2026-10-03.** Drop Arm B. The Nebius submission must then
  meet the NVIDIA-model requirement another way, decided by 2026-10-05, or not be
  submitted.
- **K5 — fewer than 100 participants per class pass §4.1.** Report Arm A as
  underpowered and issue no verdict.

## 10. Results ledger (append-only)

| Date | Arm | Result | Commit |
| --- | --- | --- | --- |

## 11. Deviations (append-only)

| Date | Section | Change | Reason |
| --- | --- | --- | --- |

## 12. Scope of claims

This is research, not clinical validation. A REPLICATED verdict means the ranking
signal carries over to ABIDE II under this pipeline. It says nothing about
diagnostic performance, screening use, or causal mechanism. Clinical claims remain
out of scope, per `CONTRIBUTING.md` and `MODEL_CARD.md`.

## 13. Timeline

| Gate | Date |
| --- | --- |
| Freeze | 2026-10-01 |
| P2 resolved | 2026-10-03 |
| Arm A verdict | 2026-10-12 |
| Arm B result | 2026-10-24 |
| Nebius × NVIDIA submission | 2026-10-30, 10:00 PDT |
