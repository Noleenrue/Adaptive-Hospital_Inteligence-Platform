# Portfolio Presentation Outline

**Audience:** Data Scientist, Machine Learning Engineer, and Python Backend Developer interviewers
**Suggested length:** 7–10 minutes
**Project framing:** Professional portfolio prototype; not a university assignment.

> **Research prototype — not clinically validated.** The model estimates 30-day readmission risk **at hospital discharge**, not at admission. It is not a clinical decision tool.

## Slide 1 — The project and the problem

- Adaptive Hospital Intelligence Platform: model, API, database, dashboard, explainability.
- Problem framed as estimating 30-day readmission risk after an eligible discharge.
- State early that this is research-only and not clinically validated.
- Emphasize portfolio goals: demonstrate end-to-end engineering and ML judgment, not claim clinical impact.

**Role emphasis:** Data Scientist — prediction target and evaluation framing; ML Engineer — lifecycle and artifact integrity; Backend Developer — service and persistence architecture.

## Slide 2 — Dataset and prediction-time contract

- UCI Diabetes 130-US Hospitals for 1999–2008; 101,766 source encounters.
- 1,652 expired encounters excluded; corrected-v3 cohort: 100,114 encounters and 70,439 patients.
- Intended prediction point is discharge. Disposition, length of stay, and completed encounter information are not admission-time inputs.
- Sensitive demographic attributes and single historical source require careful generalization assessment.
- Cite UCI DOI 10.24432/C5230J and CC BY 4.0 attribution.

## Slide 3 — Data pipeline and features

- Adapter/canonical fields, preprocessing, diagnosis categories, prior-utilization features.
- Ten-year age bucket midpoint is approximate.
- HbA1c and glucose use explicit `Not measured` categories to preserve missingness meaning.
- Pipeline preprocessing is fit on training data.

**Key engineering point:** training-serving parity requires the same input mapping, column order, and missing-value semantics.

## Slide 4 — Split strategy and model

- GroupShuffleSplit by patient ID, seed 42.
- Train: 63,949 encounters / 45,080 patients.
- Calibration: 16,090 / 11,271.
- Test: 20,075 / 14,088.
- XGBoost pipeline with imbalance weighting; separate Platt calibrator fitted on the calibration partition.
- Test partition not used to fit the model/calibrator or tune thresholds.

## Slide 5 — Corrected-v3 evaluation

Use only values in the [corrected-v3 report](../reports/corrected_v3_model_evaluation_report.md):

- ROC-AUC 0.6656 (patient-cluster bootstrap 95% CI 0.6528–0.6776).
- PR-AUC 0.2232 (0.2038–0.2439); test prevalence 11.48%.
- Brier 0.0972; log loss 0.3374; ECE 0.0033.
- At the unchanged 0.15 threshold: accuracy 0.7628, precision 0.2161, recall 0.4056, F1 0.2820.
- Explain this is one patient-disjoint random split—not temporal or external validation.

## Slide 6 — Calibration and risk categories

- Distinguish raw XGBoost probability from Platt-calibrated probability.
- Held-out calibration diagnostics: slope 1.0089 (95% CI 0.9346–1.0833), intercept 0.0333 (−0.1287–0.1880).
- Existing tiers: Low `<0.10`, Moderate `0.10–<0.15`, High `≥0.15`.
- Test observed rates: 7.04%, 12.08%, and 21.61%, respectively.
- 32.28% of test readmissions were in Low risk: Low does not mean zero risk.
- Tiers are not validated clinical decision rules.

## Slide 7 — Explainability and fidelity

- Native XGBoost TreeSHAP on exact pipeline-transformed input.
- Preserve feature ordering and sparse missing-value semantics; contributions reconstruct raw margin within tolerance.
- Contributions are raw-margin/log-odds values, not calibrated probability percentage points.
- SHAP is not causal and does not prove a feature caused an outcome.

## Slide 8 — Application architecture

- Streamlit → FastAPI → predictor + calibrator.
- `GET /health` reports loaded model identity.
- `POST /predict` produces result and inserts PostgreSQL history record.
- `POST /explain` computes SHAP without persistence.
- History GET routes read saved records.
- UI submission lock is not API/database idempotency.

## Slide 9 — Testing and engineering lessons

- Tests cover split/provenance guards, metrics/bootstrap, artifacts, mappings, risk policy, SHAP fidelity, and mocked dashboard/API behavior.
- Keep database-writing tests isolated from mocked safe tests.
- Main lessons: group by patient; keep calibration separate; align training and serving inputs; bind explanation to exact deployed pipeline; state persistence effects clearly.
- No CI/CD, cloud hosting, or production operations are claimed.

## Slide 10 — Limitations and next steps

- Not clinically validated; no real-world impact or prospective evaluation.
- No temporal, hospital-held-out, or external validation.
- Further assess subgroup performance/calibration, temporal drift, clinical utility, and deployment governance.
- Improve reproducibility: complete pinned dependency manifest, clean-clone artifact/data acquisition, database bootstrap/migrations, and isolated integration fixtures.
- Close by describing role-specific skills and evidence without implying patient-care readiness.

## Optional live demo

Use [the interview demo script](interview_demo.md), synthetic inputs only. Make clear before submission that `POST /predict` writes to PostgreSQL; do not submit unless using a disposable local database.
