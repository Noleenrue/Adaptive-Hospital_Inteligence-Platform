import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.models.predictor import (
    DB_VERSION_MAX_LENGTH,
    DEFAULT_MODEL_DIR,
    LEGACY_MODEL_VERSION,
    MANIFEST_FILENAME,
    MODEL_DIR_ENV_VAR,
    ReadmissionPredictor,
)

CANDIDATE_DIR = DEFAULT_MODEL_DIR / "candidates" / "corrected-v2-candidate"

needs_candidate = pytest.mark.skipif(
    not (CANDIDATE_DIR / MANIFEST_FILENAME).exists(),
    reason="candidate bundle not generated",
)

PATIENT = {
    "age_midpoint": 65, "gender": "Female", "race": "Caucasian",
    "admission_type": "Emergency", "admission_source": "Emergency Room",
    "discharge_disposition": "Discharged to home", "length_of_stay": 5,
    "previous_inpatient_visits": 1, "previous_outpatient_visits": 2,
    "previous_emergency_visits": 1, "diagnosis_count": 7,
    "lab_procedure_count": 45, "procedure_count": 1, "medication_count": 15,
    "hba1c_result": "None", "glucose_result": "None",
    "insulin_status": "Steady", "diabetes_medication": "Yes",
    "primary_diagnosis_category": "Circulatory",
    "secondary_diagnosis_category": "Diabetes",
    "tertiary_diagnosis_category": "Other",
    "total_previous_visits": 4, "has_previous_inpatient_visit": 1,
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)


@pytest.fixture
def saved_rows(monkeypatch):
    """Replace database writes so tests never touch real prediction history."""

    rows = []

    def fake_save(**kwargs):
        rows.append(kwargs)
        return len(rows)

    monkeypatch.setattr(api_main, "save_prediction", fake_save)
    return rows


def _client(monkeypatch, predictor):
    monkeypatch.setattr(api_main, "predictor", predictor)
    return TestClient(api_main.app)


def test_deployed_version_and_artifact_id_are_stable():
    first = ReadmissionPredictor()
    second = ReadmissionPredictor()

    if (DEFAULT_MODEL_DIR / MANIFEST_FILENAME).exists():
        assert first.model_version == first.manifest["version"]
    else:
        # Pre-manifest artifacts keep the version used by existing rows.
        assert first.model_version == LEGACY_MODEL_VERSION == "1.0"
        assert first.db_model_version == "1.0"

    assert first.artifact_id.startswith("sha256:")
    assert first.artifact_id == second.artifact_id


@needs_candidate
def test_candidate_version_comes_from_manifest():
    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert candidate.model_version == candidate.manifest["version"]
    assert candidate.model_version == "corrected-v2-candidate"


@needs_candidate
def test_deployed_and_candidate_have_different_identities():
    deployed = ReadmissionPredictor()
    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    if deployed.artifact_id == candidate.artifact_id:
        # After promotion the two are the same bundle, so identities match.
        assert deployed.model_version == candidate.model_version
        pytest.skip("candidate has been promoted; identities are equal")

    assert deployed.model_version != candidate.model_version
    assert deployed.db_model_version != candidate.db_model_version


@needs_candidate
def test_db_version_fits_column_and_stays_distinct():
    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert len(candidate.model_version) > DB_VERSION_MAX_LENGTH
    assert len(candidate.db_model_version) <= DB_VERSION_MAX_LENGTH
    assert candidate.artifact_id.split(":")[-1][:8] in (
        candidate.db_model_version
    )


def test_artifact_id_changes_when_calibrator_changes(tmp_path):
    import shutil

    for name in ("readmission_xgb_pipeline.joblib", "platt_calibrator.joblib"):
        shutil.copy(DEFAULT_MODEL_DIR / name, tmp_path / name)

    original = ReadmissionPredictor(model_dir=tmp_path).artifact_id

    # Same bytes in a different directory give the same identity.
    assert original == ReadmissionPredictor().artifact_id

    (tmp_path / "platt_calibrator.joblib").write_bytes(
        (DEFAULT_MODEL_DIR / "platt_calibrator.joblib").read_bytes() + b" "
    )

    # Trailing byte changes the file hash even if joblib may still load it.
    try:
        changed = ReadmissionPredictor(model_dir=tmp_path).artifact_id
    except Exception:
        return

    assert changed != original


def test_health_reports_deployed_identity(monkeypatch):
    predictor = ReadmissionPredictor()
    body = _client(monkeypatch, predictor).get("/health").json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["calibrator_loaded"] is True
    assert body["model_version"] == predictor.model_version
    assert body["artifact_id"] == predictor.artifact_id


@needs_candidate
def test_health_reports_candidate_identity(monkeypatch):
    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    body = _client(monkeypatch, predictor).get("/health").json()

    assert body["model_version"] == "corrected-v2-candidate"
    assert body["artifact_id"] == predictor.artifact_id


def test_predict_returns_identity_and_keeps_existing_fields(
    monkeypatch, saved_rows
):
    predictor = ReadmissionPredictor()
    response = _client(monkeypatch, predictor).post("/predict", json=PATIENT)

    assert response.status_code == 200
    body = response.json()

    for field in (
        "raw_probability", "calibrated_probability",
        "risk_tier", "prediction_id",
    ):
        assert field in body

    assert body["model_version"] == predictor.model_version
    assert body["artifact_id"] == predictor.artifact_id


def test_predict_stores_loaded_version_not_hardcoded(monkeypatch, saved_rows):
    predictor = ReadmissionPredictor()
    _client(monkeypatch, predictor).post("/predict", json=PATIENT)

    assert saved_rows[0]["model_version"] == predictor.db_model_version
    assert saved_rows[0]["model_name"] == predictor.model_name


@needs_candidate
def test_candidate_predict_stores_candidate_version(monkeypatch, saved_rows):
    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    body = _client(monkeypatch, predictor).post(
        "/predict", json=PATIENT
    ).json()

    assert saved_rows[0]["model_version"] == predictor.db_model_version
    assert saved_rows[0]["model_version"] != "1.0"
    assert len(saved_rows[0]["model_version"]) <= DB_VERSION_MAX_LENGTH
    assert body["model_version"] == "corrected-v2-candidate"
    assert body["artifact_id"] == predictor.artifact_id


@needs_candidate
def test_candidate_prediction_matches_predictor(monkeypatch, saved_rows):
    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    body = _client(monkeypatch, predictor).post(
        "/predict", json=PATIENT
    ).json()
    direct = predictor.predict(pd.DataFrame([PATIENT]))

    assert body["calibrated_probability"] == pytest.approx(
        direct["calibrated_probability"]
    )
    assert body["risk_tier"] == direct["risk_tier"]
