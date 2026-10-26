# corrected-v3 Model Quality and Reliability Evaluation

**Model:** `corrected-v3`
**Artifact ID:** `sha256:21438015f642839e`
**Evaluation population:** held-out test set, 20,075 encounters from 14,088 patients
**Intended prediction time:** **at hospital discharge**
**Clinical status:** Research prototype; **not clinically validated**.

## Evaluation provenance

The deployed model and calibrator were loaded without modification. Their identity, manifest file hashes, dataset SHA-256, row counts, expired-discharge exclusion, deterministic random seed (42), split sizes, patient counts, and pairwise patient separation were checked against the promoted corrected-v3 manifest before scoring. The data hash was `66cd14a7d3d634e6182f63e78fd307bbbe9264c997de73439caf283b757cd28a`.

The dataset contains 101,766 rows before filtering. Encounters whose `discharge_disposition` contains “Expired” were excluded before splitting (1,652 excluded; 100,114 eligible). GroupShuffleSplit on `patient_id` produced 63,949 train / 16,090 calibration / 20,075 test encounters. Patient counts were 45,080 / 11,271 / 14,088. The test partition alone was scored. No predictive model or production calibrator was fit or recalibrated, and no threshold tuning used test outcomes. Calibration slope/intercept below are diagnostic statistics only.

## Test-set performance

Threshold-dependent measures use the existing intervention threshold **0.15**; thresholds were not changed.

| Metric | Test result | Patient-cluster bootstrap 95% CI |
|---|---:|---:|
| ROC-AUC | 0.6656 | 0.6528–0.6776 (valid 500) |
| PR-AUC (Average Precision) | 0.2232 | 0.2038–0.2439 (valid 500) |
| Accuracy | 0.7628 | 0.7561–0.7701 (valid 500) |
| Precision | 0.2161 | 0.2009–0.2300 (valid 500) |
| Recall / sensitivity | 0.4056 | 0.3821–0.4280 (valid 500) |
| F1-score | 0.2820 | 0.2644–0.2976 (valid 500) |
| Brier score | 0.0972 | 0.0933–0.1004 (valid 500) |
| Log loss | 0.3374 | 0.3270–0.3462 (valid 500) |
| ECE (10 equal-width bins) | 0.0033 | 0.0019–0.0084 (valid 500) |

Confusion matrix at threshold 0.15 (rows actual [no readmission, readmitted], columns predicted [below threshold, at/above threshold]): **[[14379, 3391], [1370, 935]]**.

The bootstrap sampled patients with replacement and retained all encounters for each sampled patient. It requested 500 resamples, of which 500 contained both outcome classes and were used; 0 single-class resamples were skipped. Per-metric valid counts are in the JSON output.

## Calibration

Calibration slope: **1.0089** (95% CI 0.9346–1.0833 (valid 500)).
Calibration intercept on the logit scale: **0.0333** (95% CI -0.1287–0.1880 (valid 500)).
Mean predicted risk: **0.1133**; observed readmission rate: **0.1148**.

The slope and intercept are diagnostic logistic recalibration regression estimates computed on the held-out predictions and outcomes. They were not applied to predictions and did not modify the Platt calibrator. The reliability diagram and equal-width / quantile-bin tables are saved alongside this report. These are single-split estimates and do not establish calibration in future or different populations.

## Prediction-time availability and limitations

This is a **discharge-time readmission prediction model**, not an admission-time model. Discharge disposition and length of stay are only known at or near discharge. Final diagnoses and encounter-completion details—including current-stay laboratory results, procedure counts, and medication information—may be unavailable or incomplete earlier. Using these inputs for an earlier prediction would create a feature-availability mismatch and could constitute temporal leakage. The intended scoring workflow must enforce the discharge-time contract.

Death-related discharges are excluded because a patient who dies during the encounter is not eligible for a post-discharge readmission prediction. This does not exclude hospice discharges. The model uses weighted XGBoost training for class imbalance and a previously fitted Platt calibrator; neither was refit during this evaluation.

This evaluation uses one patient-disjoint random split of a single dataset. It is not temporal, hospital-held-out, or external validation. Performance may differ across hospitals, time periods, coding systems, and populations. No clinical utility, fairness, or prospective impact is established. The model is **not clinically validated** and should not be used as a standalone clinical decision.

## Versioning note

Existing generic `evaluation_*` outputs and `model_evaluation_report.md` are retained as historical corrected-v2 results; they were not overwritten or used as corrected-v3 results. The v3-versus-v2 comparison file contains both versions and was used only to confirm provenance; metrics above are calculated from the loaded corrected-v3 bundle and its verified test split.
