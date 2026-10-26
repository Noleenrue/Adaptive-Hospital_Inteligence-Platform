# SHAP Fidelity and HbA1c/Glucose Encoding Fixes

Date: 2026-10-22. Scope: concerns 1 and 3 from [model_evaluation_report.md](model_evaluation_report.md).
Deployed artifacts (`models/`), thresholds, notebooks, the dataset and PostgreSQL records were not modified.

## 1. SHAP explanation fidelity

### Problem (reproduced before the fix)
`ReadmissionSHAPExplainer` converted the preprocessor output to a dense array and used `shap.TreeExplainer`.
The deployed pipeline predicts from the sparse matrix, where XGBoost treats absent entries as *missing*, so
the explainer described a different model input.

| Case (deployed model) | Deployed raw margin | Old explainer reconstruction | Gap |
|---|---|---|---|
| Sample patient | +0.1444 | -2.6164 | -2.76 |
| No prior visits | -0.3372 | -2.4039 | -2.07 |
| Age 85, 6 prior inpatient | +1.2200 | -2.1607 | -3.38 |
| HbA1c >8, glucose >300 | +0.2151 | -2.6164 | -2.83 |

Across 2,000 test encounters (pre-fix evaluation): margin differed by more than 0.1 in 99.75% of cases (mean 2.43,
maximum 4.82), top-10 feature overlap with the correct values was 3.4%.

### Fix
`src/explainability/shap_explainer.py` now calls `booster.predict(xgb.DMatrix(Z), pred_contribs=True)` on the exact
pipeline output `Z` (not densified), after the same `prepare_features` input mapping that `predict` uses. Every call
checks that contributions plus base value equal `classifier.predict(Z, output_margin=True)` (tolerance 1e-3) and raises otherwise.
`explain()` additionally returns `base_value`, `margin`, `raw_probability`, `model_version` and `artifact_id`.
`/explain` now uses the API's own loaded predictor (previously it built a new one per request) and returns the same
fields additively; existing keys are unchanged.

### Verification (after the fix)
- 300 test encounters, deployed model: maximum reconstruction error 1.1e-6; candidate 1.9e-6.
- `evaluate_model.py`: 200 rows reconstruct the margin and equal the batch native SHAP (all 19 integrity checks pass).
- End-to-end through the FastAPI app: `sigmoid(margin)` equals `/predict` `raw_probability` to 1e-5 for every case.
- Feature names follow `preprocessor.get_feature_names_out()` order; values match the transformed row.

### Remaining SHAP caveats
- Values are on the raw log-odds scale, not calibrated probability.
- Under XGBoost's sparse semantics, a zero numeric value (e.g. 0 prior visits) and an absent one-hot category are *missing*, which is how the deployed model was trained and scored. This is faithful to the model, but the "feature value 0" shown for such features means "missing" to the trees.
- Correlated features share credit; SHAP is not causal.
- Notebook 04 still explains the legacy `readmission_xgboost_v1` model and was not changed.

## 2. HbA1c / glucose training-serving mismatch

### Problem (reproduced before the fix)
In the training CSV, "not measured" is blank (83.1% HbA1c, 94.8% glucose). The training imputer filled blanks with the
most frequent *measured* value. Serving sent the literal "None", which the encoder maps to all zeros.

| Input to deployed model | Encoded as |
|---|---|
| blank (as trained) | `hba1c_result_>8` = 1, `glucose_result_Norm` = 1 |
| "None" (as the dashboard sends) | all zeros |

Impact on the 20,075 test encounters (deployed model): mean absolute change in calibrated probability 0.0046
(maximum 0.061); 5.5% of encounters change risk tier.

### Fix (no change to clinical meaning)
`src/preprocessing/lab_encoding.py` maps every "not measured" spelling (blank, NaN, "None", "none", "NaN", "null",
"Not measured") to one explicit category, `Not measured`. Measured values (`Norm`, `>7`, `>8`, `>200`, `>300`) and all other columns are never changed.
The mapping is applied identically in training (`scripts/export_model_bundle.py --lab-not-measured`) and in serving
(`ReadmissionPredictor.prepare_features`, used by both `predict` and the explainer).

The mapping is opt-in through the manifest key `input_mapping: "lab-not-measured-v1"`. Artifacts without it (the
currently deployed bundle) behave exactly as before, so deployed predictions do not change silently. The predictor
refuses to load a bundle that declares the mapping if its encoder lacks the `Not measured` category.

### Retraining required: yes
The deployed encoder has no `Not measured` category, and the trained trees learned blanks as `>8` / `Norm`. The
mapping therefore cannot be applied to the deployed model without changing its behaviour. A new candidate was
trained on the identical patient split and seeds, with a calibrator refitted on the calibration partition only.
No test data was used for fitting, calibration or thresholds (thresholds unchanged: 0.10 / 0.15 / 0.15).

### Candidate: `models/candidates/corrected-v3` (not promoted)
Version `corrected-v3`, artifact `sha256:21438015f642839e`, hash-verified manifest. Same split: 63,949 / 16,090 / 20,075 encounters.

| Metric (test, n=20,075) | Deployed v2 | Candidate v3 |
|---|---|---|
| ROC-AUC | 0.6655 | 0.6656 |
| PR-AUC | 0.2230 | 0.2232 |
| Brier | 0.0971 | 0.0972 |
| Log loss | 0.3374 | 0.3374 |
| ECE (10 equal-width bins) | 0.0048 | 0.0033 |
| Blank vs "None" skew (mean abs prob. diff) | 0.0046 | 0.0 |

| Tier | v2 n / observed / mean pred. | v3 n / observed / mean pred. |
|---|---|---|
| Low | 10,630 / 0.0716 / 0.0679 | 10,568 / 0.0704 / 0.0678 |
| Moderate | 5,115 / 0.1183 / 0.1224 | 5,181 / 0.1208 / 0.1223 |
| High | 4,330 / 0.2169 / 0.2138 | 4,326 / 0.2161 / 0.2134 |

v3 versus v2 on identical encounters: mean absolute probability difference 0.0053 (maximum 0.138); 5.9% of encounters change tier.
Source: `candidate_corrected-v3_comparison.json` (regenerate with `python -m src.evaluation.compare_candidate`).

## 3. Tests (safe suite; DB tests deselected)

| | Before | After |
|---|---|---|
| Passed | 55 | 89 |
| Skipped | 1 | 1 |
| Expected failure (documented) | 0 | 1 |
| Failed | 0 | 0 |

New: `tests/test_explanation_fidelity.py` (margin reconstruction, name order, value alignment, batch/single scoring
agreement, real rows) and `tests/test_lab_input_mapping.py` (mapping rules, training/serving parity, candidate
validation). The one expected failure documents that the *deployed* model still has the blank-vs-"None" mismatch; it
disappears when v3 is promoted.

## 4. Files changed
- Modified: `src/explainability/shap_explainer.py`, `src/models/predictor.py`, `api/main.py`,
  `scripts/export_model_bundle.py`, `src/evaluation/evaluate_model.py`, `reports/model_evaluation_report.md`
- New: `src/preprocessing/lab_encoding.py`, `src/evaluation/compare_candidate.py`, `tests/test_explanation_fidelity.py`,
  `tests/test_lab_input_mapping.py`, `models/candidates/corrected-v3/` (candidate only),
  `reports/candidate_corrected-v3_comparison.json`, this file
- Regenerated by the read-only evaluation script: `reports/evaluation_*`

## 5. Not done / needs approval
- Promotion of `corrected-v3` into `models/` (use `scripts/promote_candidate.py`, which backs up and rolls back on failure).
- A real PostgreSQL insert or live dashboard smoke test for v3.
- Re-running notebook 04 on the corrected model.
