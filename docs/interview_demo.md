# Interview Demo: 5–10 Minute Walkthrough

## Safety and demo prerequisites

This walkthrough uses **synthetic data only**. The application is a research prototype, not clinically validated, and estimates 30-day readmission risk at **hospital discharge**, not at admission. It is not for clinical decisions.

The API's `POST /predict` route inserts a durable PostgreSQL prediction record. Use a disposable local database and synthetic sample values. Do not use real patient data or a shared/production database. `POST /explain`, `GET /health`, and history GET requests do not create prediction records.

Before the interview, confirm that local model artifacts and processed data are available, install dependencies (see the README caveats), configure a local PostgreSQL database and required `DB_*` environment variables, and apply `database/schema.sql`. Do not display `.env`, credentials, or patient identifiers during screen sharing.

## Start the local services

Open two PowerShell terminals at the repository root. In the first:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

In the second:

```powershell
python -m streamlit run dashboard/app.py --server.port 8502
```

Open the dashboard at `http://127.0.0.1:8502`. Its default API URL is `http://127.0.0.1:8001`; if needed, configure `API_URL` before starting Streamlit. Avoid changing the URL to an older or unverified API.

## Timed script

### 0:00–1:00 — Frame the project

**Show:** README overview and the dashboard Overview page.

**Say:** “This is a professional portfolio prototype demonstrating a healthcare ML workflow and API-backed application. It estimates 30-day readmission risk at discharge. It is research-only, not clinically validated, and not a clinical decision tool.”

### 1:00–2:00 — Confirm the served model identity

**Show:** Dashboard Overview, including the version and artifact identity fetched from `/health`. Optionally open `http://127.0.0.1:8001/docs` in a browser (read-only interactive docs page; do not execute write routes).

**Say:** “The dashboard reads model version and artifact ID from the running API rather than hardcoding them. The corrected-v3 evaluation is tied to a patient-disjoint held-out test partition.”

Expected verified identity: `corrected-v3`, `sha256:21438015f642839e`. If the health page reports a different identity, stop and describe the environment accurately rather than claiming it is v3.

### 2:00–4:00 — Explain verified model evaluation

**Show:** [Corrected-v3 evaluation report](../reports/corrected_v3_model_evaluation_report.md) or its reliability diagram.

**Say:** “The test partition contains 20,075 encounters from 14,088 patients. ROC-AUC is 0.6656; PR-AUC is 0.2232. At the existing 0.15 threshold, recall is 0.4056 and precision is 0.2161. These are single-dataset, single-split results—not external validation.”

Clarify probability meanings: “The XGBoost raw probability and calibrated probability are separate. Platt scaling produces the calibrated estimate; the held-out calibration slope and intercept are diagnostics, not another fitted calibration step.”

### 4:00–6:00 — Demonstrate a synthetic assessment

**Show:** Patient Risk Assessment using the generated synthetic sample or synthetic values. The repository includes `scripts/_sample_patient.json`; do not substitute real patient details.

Before submitting, point out that the form is intended for discharge-time inputs and explain the selected inputs are synthetic. Show the calibrated probability as the primary result, raw model probability separately, and the research tier.

**Database effect:** Submitting the form triggers `POST /predict` and creates a PostgreSQL record. Only do this against a disposable local database after confirming it is safe to write. A timeout may occur after a write; the UI's lock reduces accidental repeat submissions but does not guarantee exactly-once persistence. If the demo environment is not safe to write to, do not submit; explain the flow from code instead.

### 6:00–7:30 — Demonstrate SHAP explanation

**Show:** Explanation for the same synthetic assessment.

**Say:** “This uses the deployed pipeline transformation and native XGBoost TreeSHAP contributions. Contributions reconstruct the raw model margin; they are not percentage-point changes in calibrated probability and are not causal explanations.”

`POST /explain` computes an explanation without inserting a prediction row.

### 7:30–8:30 — Demonstrate Prediction History

**Show:** Prediction History after the safe synthetic submission, or the page without submitting if a writable disposable database is unavailable.

**Say:** “History reads saved records using GET; refreshing this page does not create predictions. The prediction write occurs on `POST /predict`, not on history retrieval.”

### 8:30–9:30 — Close with limitations and engineering choices

**Show:** [Architecture](architecture.md) or the case study.

Highlight patient-grouped splits, separate calibration data, explicit unmeasured-lab handling, artifact hashes, safe error handling, mocked dashboard tests, and the need for temporal/external validation before any consideration of clinical use.

## Endpoint and persistence quick reference

| Operation | Method and route | Persists a prediction? |
|---|---|---|
| Health/model identity | `GET /health` | No |
| Prediction | `POST /predict` | **Yes — INSERT into PostgreSQL** |
| SHAP explanation | `POST /explain` | No |
| History list | `GET /predictions` | No |
| History detail | `GET /predictions/{prediction_id}` | No |

## Screenshot checklist

No dashboard screenshots are currently supplied. The entries below are **placeholder filenames only**, not existing screenshots or images. If capturing them later, use synthetic values and produce:

- `docs/assets/overview.png`
- `docs/assets/assessment.png`
- `docs/assets/explanation.png`
- `docs/assets/history.png`

Review every image for credentials, local paths, secrets, and real patient data before publishing.
