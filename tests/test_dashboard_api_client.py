import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import api_client  # noqa: E402
from api_client import (  # noqa: E402
    DashboardAPIError,
    get_health,
    get_history,
    post_prediction,
    validate_patient_data,
)


def _response(payload, status=200):
    response = Mock()
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "calibrator_loaded": True,
        "model_version": "corrected-v3",
        "artifact_id": "sha256:21438015f642839e",
    }


def _patient():
    return {
        "age_midpoint": 65.0,
        "gender": "Female",
        "race": "Caucasian",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "discharge_disposition": "Discharged to home",
        "length_of_stay": 4,
        "previous_inpatient_visits": 1,
        "previous_outpatient_visits": 0,
        "previous_emergency_visits": 0,
        "diagnosis_count": 5,
        "lab_procedure_count": 40,
        "procedure_count": 1,
        "medication_count": 12,
        "hba1c_result": "None",
        "glucose_result": "None",
        "insulin_status": "Steady",
        "diabetes_medication": "Yes",
        "primary_diagnosis_category": "Circulatory",
        "secondary_diagnosis_category": "Diabetes",
        "tertiary_diagnosis_category": "Other",
        "total_previous_visits": 1,
        "has_previous_inpatient_visit": 1,
    }


def test_dashboard_defaults_to_corrected_v3_api_port():
    assert api_client.DEFAULT_API_URL == "http://127.0.0.1:8001"
    assert api_client.API_URL != "http://127.0.0.1:8000"


def test_health_uses_api_reported_model_identity(monkeypatch):
    get = Mock(return_value=_response(_health()))
    monkeypatch.setattr(api_client.requests, "get", get)

    result = get_health()

    assert result["model_version"] == "corrected-v3"
    assert result["artifact_id"] == "sha256:21438015f642839e"
    get.assert_called_once_with(
        f"{api_client.API_URL}/health", timeout=10
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**_health(), "model_loaded": False},
        {**_health(), "artifact_id": "not-an-artifact"},
        {**_health(), "artifact_id": "sha256:"},
    ],
)
def test_health_rejects_unverified_model_identity(monkeypatch, payload):
    monkeypatch.setattr(
        api_client.requests, "get", Mock(return_value=_response(payload))
    )

    with pytest.raises(DashboardAPIError, match="did not confirm"):
        get_health()


def test_connection_error_is_clear_and_does_not_expose_exception(
    monkeypatch,
):
    secret_detail = "secret-path-and-patient-data"
    monkeypatch.setattr(
        api_client, "API_URL", "http://user:api-secret@localhost:8001"
    )
    monkeypatch.setattr(
        api_client.requests,
        "get",
        Mock(side_effect=requests.ConnectionError(secret_detail)),
    )

    with pytest.raises(DashboardAPIError) as exc:
        get_health()

    assert "configured API" in str(exc.value)
    assert "api-secret" not in str(exc.value)
    assert secret_detail not in str(exc.value)


def test_invalid_input_response_does_not_expose_server_body(monkeypatch):
    response = Mock()
    response.status_code = 422
    error = requests.HTTPError("unsafe detail")
    error.response = response
    monkeypatch.setattr(
        api_client.requests, "post", Mock(side_effect=error)
    )

    with pytest.raises(DashboardAPIError) as exc:
        post_prediction(_patient())

    assert "submitted data is invalid" in str(exc.value)
    assert "unsafe detail" not in str(exc.value)


def test_history_uses_get_only_and_does_not_submit_prediction(monkeypatch):
    get = Mock(return_value=_response({"total": 0, "predictions": []}))
    post = Mock()
    monkeypatch.setattr(api_client.requests, "get", get)
    monkeypatch.setattr(api_client.requests, "post", post)

    assert get_history() == {"total": 0, "predictions": []}
    get.assert_called_once_with(
        f"{api_client.API_URL}/predictions",
        timeout=10,
        params={"limit": 100},
    )
    post.assert_not_called()


def test_prediction_response_returns_api_values_without_mutating_input(
    monkeypatch,
):
    patient = _patient()
    original = patient.copy()
    payload = {
        "raw_probability": 0.45,
        "calibrated_probability": 0.097,
        "risk_tier": "Low",
        "prediction_id": 42,
        "model_version": "corrected-v3",
        "artifact_id": "sha256:21438015f642839e",
    }
    post = Mock(return_value=_response(payload))
    monkeypatch.setattr(api_client.requests, "post", post)

    result = post_prediction(patient)

    assert result == payload
    assert patient == original
    post.assert_called_once_with(
        f"{api_client.API_URL}/predict",
        timeout=30,
        json=patient,
    )


def test_patient_validation_rejects_blank_required_text():
    patient = _patient()
    patient["discharge_disposition"] = " "

    assert "required text fields" in validate_patient_data(patient)


def test_patient_validation_rejects_out_of_range_values():
    patient = _patient()
    patient["age_midpoint"] = 121

    assert "outside the supported range" in validate_patient_data(patient)


def test_patient_validation_accepts_complete_payload():
    assert validate_patient_data(_patient()) is None
