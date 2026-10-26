# Technical Case Study: Adaptive Hospital Intelligence Platform

## Portfolio context

The Adaptive Hospital Intelligence Platform is a professional software and machine-learning portfolio project for Data Scientist, Machine Learning Engineer, and Python Backend Developer roles. It demonstrates a path from a historical healthcare dataset to a model-backed API and interactive dashboard. It is not a university assignment, a clinical product, or evidence of real-world clinical impact.

> **Research prototype — not clinically validated.** The model estimates 30-day readmission risk **at hospital discharge**, not at admission. Risk tiers are not validated clinical decision rules, and Low risk does not mean zero risk.

## Problem and intended use

Hospital readmission risk prediction is a binary classification task: estimate whether an eligible encounter will be followed by readmission within 30 days. The input contract is discharge-time because some features (including discharge disposition and length of stay) are only known at or near discharge. The system demonstrates how a calibrated model can be served, inspected, and explained in a local application.

This project does not identify an intervention that will prevent readmission. It has no prospective evaluation, clinical workflow validation, or evidence that use of its outputs improves outcomes.

## Dataset and target

The project uses the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset. UCI describes records from 130 US hospitals and integrated delivery networks; the dataset includes 101,766 encounters and demographic, admission, utilization, diagnosis, laboratory, medication, and outcome fields. Its target concerns readmission within 30 days after discharge.

The corrected-v3 manifest/report record 1,652 death-related discharges excluded before partitioning, leaving 100,114 eligible encounters from 70,439 patients. Death-related discharges are excluded because a person who dies during the index encounter is not at risk of a subsequent readmission. Hospice discharges are not described as excluded by this rule.

**Attribution and license:** UCI lists the dataset under CC BY 4.0. Cite Clore, Cios, DeShazo, and Strack (2014), *Diabetes 130-US Hospitals for Years 1999–2008*, UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J). The associated study is Strack et al. (2014), “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” *BioMed Research International*, Article 781670, https://doi.org/10.1155/2014/781670. Confirm current terms and required attribution on the [official UCI record](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) when redistributing data or derived assets.

## Data preparation and feature engineering

The implementation is organized around adapters, preprocessing, and feature building under `src/adapters/`, `src/preprocessing/`, and `src/features/`. The corrected-v3 model consumes 23 features listed in `models/manifest.json`, including demographics, admission/discharge information, length of stay, prior utilization, diagnosis categories, laboratory result categories, medication information, and derived utilization features.

Important transformations include:

- Age is represented by the midpoint of the supplied ten-year age band; it is an approximation, not exact age.
- Diagnosis codes are mapped to broad ICD-9 categories.
- Prior inpatient, outpatient, and emergency visit counts are retained; total prior visits and an inpatient-visit indicator are also used.
- HbA1c/glucose missingness is encoded consistently as the explicit category `Not measured` for the corrected-v3 artifact. This preserves the distinction between an unmeasured result and an observed result; it does not impute an unmeasured value as normal.
- Numeric imputation and categorical encoding are fitted within the model pipeline on training data.

This design reduces training-serving mapping drift. The served `ReadmissionPredictor` applies the artifact-declared mapping for both prediction and explanation inputs.

## Partitioning and modeling

The corrected-v3 workflow excludes expired encounters and then uses `GroupShuffleSplit` by `patient_id`, seed 42. The 80% development portion is split again to create a separate calibration partition. The manifest and evaluation report record:

| Partition | Encounters | Patients | Purpose |
|---|---:|---:|---|
| Train | 63,949 | 45,080 | Fit preprocessing and XGBoost |
| Calibration | 16,090 | 11,271 | Fit the probability calibrator |
| Test | 20,075 | 14,088 | Final held-out evaluation |

The partitions are patient-disjoint. In particular, encounters from one patient are not distributed across train, calibration, and test. Preprocessing and the predictive model are fit on training data; Platt scaling is fit on the separate calibration data. The held-out test partition is used for final assessment, not model fitting, calibration, or threshold selection.

The model is an XGBoost binary classifier wrapped in a scikit-learn pipeline. The manifest records 138 transformed features and class-imbalance weighting (`scale_pos_weight` approximately 7.8633). The production artifact is version `corrected-v3`, artifact ID `sha256:21438015f642839e`.

## Probability calibration and risk tiers

Corrected-v3 applies a separately fitted Platt calibrator to the XGBoost raw probability. These outputs have different meanings:

- **Raw model probability:** XGBoost's uncalibrated probability output.
- **Calibrated probability:** Platt-scaled probability presented as the primary risk estimate.

The existing risk policy assigns Low below 0.10, Moderate from 0.10 to below 0.15, and High at or above 0.15. The intervention threshold is 0.15. These are research risk categories, not clinically validated decision rules.

Held-out test tier summary:

| Tier | Encounters | Mean calibrated risk | Observed readmission rate |
|---|---:|---:|---:|
| Low | 10,568 | 6.78% | 7.04% |
| Moderate | 5,181 | 12.23% | 12.08% |
| High | 4,326 | 21.34% | 21.61% |

Observed rates increase across these groups in this test partition. However, 744 of the test readmissions (32.28%) fell into Low risk. Low must not be interpreted as no risk or as a reason to withhold care.

## Evaluation results

The authoritative source is [the corrected-v3 model evaluation report](../reports/corrected_v3_model_evaluation_report.md) and its corrected-v3-specific outputs. The evaluation procedure verified artifact identity and hashes, source-data hash, filter counts, seed, split method, partition sizes, and patient separation before scoring the held-out test partition.

| Metric | Result | Patient-cluster bootstrap 95% CI |
|---|---:|---:|
| ROC-AUC | 0.6656 | 0.6528–0.6776 |
| PR-AUC / Average Precision | 0.2232 | 0.2038–0.2439 |
| Accuracy at 0.15 | 0.7628 | 0.7561–0.7701 |
| Precision at 0.15 | 0.2161 | 0.2009–0.2300 |
| Recall / sensitivity at 0.15 | 0.4056 | 0.3821–0.4280 |
| F1 at 0.15 | 0.2820 | 0.2644–0.2976 |
| Brier score | 0.0972 | 0.0933–0.1004 |
| Log loss | 0.3374 | 0.3270–0.3462 |
| ECE, 10 equal-width bins | 0.0033 | 0.0019–0.0084 |

The confusion matrix at 0.15 is `[[14379, 3391], [1370, 935]]`, with actual labels as rows (no readmission, readmitted) and predicted labels as columns (below threshold, at/above threshold). Test prevalence was 11.48%; the mean calibrated prediction was 11.33%.

Calibration diagnostics on held-out predictions were slope **1.0089** (95% CI 0.9346–1.0833) and logit intercept **0.0333** (−0.1287–0.1880). These are diagnostic regression estimates, not a new calibration fit applied to the model. The evaluator used 500 valid patient-cluster bootstrap resamples and skipped no single-class resamples.

These estimates describe one random patient-disjoint split from a single historical dataset. They are not temporal validation, hospital-held-out validation, external validation, evidence of fairness, or evidence of clinical utility. The model is not clinically validated.

## Explainability with SHAP

The API's explanation path uses native XGBoost `pred_contribs` on the same transformed pipeline input used for prediction. It retains the pipeline's feature order and sparse missing-value semantics and verifies that contributions plus the base value reconstruct the XGBoost raw margin within a numerical tolerance.

The displayed contribution values are in **raw margin/log-odds units**. They do not represent calibrated probability percentage points, and they do not establish causal influence. Categorical input features may appear as transformed one-hot columns.

## Application architecture and persistence

### FastAPI

`api/main.py` loads the predictor and exposes:

- `GET /health` — service/model health, model version, and artifact ID.
- `POST /predict` — validates inputs, returns raw and calibrated probabilities plus a tier, and **inserts a prediction record into PostgreSQL**.
- `POST /explain` — returns a SHAP explanation; the endpoint does not persist a prediction.
- `GET /predictions` and `GET /predictions/{prediction_id}` — retrieve saved prediction history.

Pydantic request/response structures are defined in `api/schemas.py`. API errors are returned with safe user-facing messages rather than raw server details.

### PostgreSQL

`database/schema.sql` defines the `predictions` table. `database/repository.py` contains the INSERT used by prediction submissions and the SELECT used for history. The table stores the raw and calibrated probabilities, risk tier, model identity, timestamp, and a patient reference. The dashboard's History page retrieves records; it does not itself submit a new prediction.

### Streamlit

`dashboard/app.py` provides Overview, Patient Risk Assessment, and Prediction History pages. The Overview retrieves model version and artifact identity from `/health`; the default API address is `http://127.0.0.1:8001` and can be overridden with `API_URL`.

The assessment UI presents calibrated probability as the primary result and raw probability separately. A UI lock discourages repeat submissions, but it is not API/database idempotency and cannot guarantee exactly-once writes if requests time out.

## Automated testing

The `tests/` suite covers predictor artifact loading and identity, data mapping, risk policy, SHAP reconstruction, evaluation split/metrics/bootstrap logic, and mocked API/dashboard behavior. A focused no-database-write command is provided in the README. The full suite is **not** automatically safe against an arbitrary configured database: `test_repository.py` inserts a test prediction and later deletes it, and `test_database_connection.py` connects to the configured database. Run those only with an isolated test database and explicit approval.

This test suite demonstrates software validation, not clinical validation. There is no claim here of CI/CD automation, a cloud deployment, production monitoring, or a prospective clinical trial.

## Engineering challenges and lessons

1. **Training-serving consistency matters.** Unmeasured laboratory results need an explicit shared encoding; treating blanks as normal would change their meaning.
2. **Repeated encounters require grouped splits.** Splitting by encounter alone can allow the same patient's records on both sides of evaluation and overstate generalization.
3. **Explanations must match the deployed calculation.** SHAP contributions are useful only when the explainer uses the exact model pipeline transformation, feature ordering, and missing-value semantics.
4. **A reported probability needs context.** Raw and calibrated probabilities are distinct, and calibration on one test sample does not guarantee reliability in future populations.
5. **Persistence should be explicit.** A prediction endpoint can write durable data; the UI should explain this and avoid suggesting that a client-side lock is a transactional guarantee.
6. **Artifact identity supports reproducibility.** A version and artifact ID from the serving API are more reliable than a hardcoded dashboard label.

## Reproducibility and current setup limitations

The UTF-8 `requirements.txt` now pins direct application, model, explanation, and test dependencies to versions verified in the existing Python 3.11 environment. `.env.example` documents the database, dashboard API, and optional model-directory variables using safe local values; it contains no real credentials.

The raw UCI dataset is downloaded separately into `data/raw/`, and the model binaries are not committed. Corrected-v3 serving requires the pipeline, calibrator, and manifest together under `models/`; obtain them only through an approved trusted transfer and check their manifest hashes. A new candidate can be exported under a unique `models/candidates/` directory using `scripts/export_model_bundle.py`, without replacing the promoted bundle. Candidate creation is not promotion, and a new artifact is not guaranteed to match corrected-v3 byte-for-byte.

The command `scripts/preprocess_uci_data.py` now wires the existing UCI adapter, mapping loader, clinical preprocessor, and feature engineer into a candidate-only preprocessing workflow. It writes to `data/processed/candidates/<run-id>/`, refuses to overwrite a run, and records input/output hashes plus a comparison against the historical processed CSV when available. The comparison covers schema/order, CSV-inferred dtypes, target distribution, patient and encounter identifier sets, duplicate counts, and row values aligned by encounter ID.

The preprocessing output retains expired-discharge rows, as does the historical processed CSV; the corrected-v3 exporter excludes the 1,652 expired discharges before patient-grouped splitting. Unmeasured HbA1c/glucose values remain missing in the processed dataset. The model-specific explicit `Not measured` mapping is applied downstream consistently during corrected-v3 training and serving. Candidate generation does not change or retrain the deployed model.

The 2026-10-26 local comparison produced 101,766 rows and 29 columns, matching CSV-inferred dtypes, target distribution, patient and encounter identifier sets, duplicate counts, and zero row-value mismatches. The candidate CSV SHA-256 exactly matched the existing processed file (`66cd14a7d3d634e6182f63e78fd307bbbe9264c997de73439caf283b757cd28a`). The candidate report also records raw source and mapping hashes. This demonstrates byte-for-byte processed-output agreement for the local inputs; it does not independently establish that those raw files are byte-identical to the historical source files because the corrected-v3 manifest records the processed-data hash, not historical raw-input hashes. Nor does data agreement guarantee a byte-identical model if a new model is trained. Discrepancies, if any, are reported rather than “fixed” to force agreement.

PostgreSQL setup is manual. `database/schema.sql` uses `CREATE TABLE IF NOT EXISTS`; initialize it only against the intended local database. No migration system or automated database provisioning is implemented. Focused mocked/read-only tests are documented in the README; database integration tests require a dedicated test database because repository tests perform writes.

## Future improvements

Priorities include external and temporal validation, subgroup calibration/performance analysis, a documented dataset-to-artifact acquisition workflow, complete pinned dependencies, database migrations and isolated test fixtures, stronger request-level idempotency, and an explicit deployment/security design. Any clinical use would require appropriate clinical governance and prospective validation; those capabilities are not implemented or claimed here.
