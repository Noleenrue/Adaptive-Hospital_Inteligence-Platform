"""HTTP helpers for the Streamlit dashboard's FastAPI backend."""

import os

import requests

DEFAULT_API_URL = "http://127.0.0.1:8001"
API_URL = (os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")


class DashboardAPIError(Exception):
    """A safe, user-facing error for dashboard API requests."""


def _request_error(exc, operation):
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        if status == 422:
            return DashboardAPIError(
                f"{operation} was rejected because the submitted data is "
                "invalid. Review required fields and allowed values."
            )
        return DashboardAPIError(
            f"{operation} failed (API returned HTTP {status}). "
            "Try again later or contact the platform administrator."
        )
    if isinstance(exc, requests.ConnectionError):
        return DashboardAPIError(
            "Cannot connect to the configured API service. "
            "Confirm the corrected-v3 FastAPI service is running and that "
            "API_URL points to the intended service."
        )
    if isinstance(exc, requests.Timeout):
        return DashboardAPIError(
            f"The API request for {operation.lower()} timed out. "
            "Check the service and try again."
        )
    return DashboardAPIError(
        f"The API request for {operation.lower()} could not be completed."
    )


def _request(method, path, operation, *, timeout, **kwargs):
    try:
        response = method(f"{API_URL}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise _request_error(exc, operation) from exc
    except ValueError as exc:
        raise DashboardAPIError(
            f"The API returned an invalid response for {operation.lower()}."
        ) from exc


def get_health():
    payload = _request(
        requests.get, "/health", "API health check", timeout=10
    )
    if not isinstance(payload, dict):
        raise DashboardAPIError("The API health response is invalid.")
    version = payload.get("model_version")
    artifact_id = payload.get("artifact_id")
    if (
        payload.get("status") != "healthy"
        or payload.get("model_loaded") is not True
        or payload.get("calibrator_loaded") is not True
        or not isinstance(version, str)
        or not version.strip()
        or not isinstance(artifact_id, str)
        or not artifact_id.startswith("sha256:")
        or not artifact_id.removeprefix("sha256:").strip()
    ):
        raise DashboardAPIError(
            "The configured API did not confirm a healthy model, "
            "calibrator, version, and artifact identity."
        )
    return payload


def get_history():
    return _request(
        requests.get,
        "/predictions",
        "Prediction history retrieval",
        timeout=10,
        params={"limit": 100},
    )


def post_prediction(patient_data):
    payload = _request(
        requests.post,
        "/predict",
        "Prediction",
        timeout=30,
        json=patient_data,
    )
    if not isinstance(payload, dict):
        raise DashboardAPIError("The prediction response is invalid.")
    try:
        raw_probability = float(payload["raw_probability"])
        calibrated_probability = float(payload["calibrated_probability"])
        prediction_id = int(payload["prediction_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardAPIError(
            "The API prediction response is missing required fields."
        ) from exc
    if (
        not 0 <= raw_probability <= 1
        or not 0 <= calibrated_probability <= 1
        or prediction_id < 0
        or payload.get("risk_tier") not in {"Low", "Moderate", "High"}
    ):
        raise DashboardAPIError(
            "The API prediction response contains invalid values."
        )
    return payload


def post_explanation(patient_data):
    return _request(
        requests.post,
        "/explain",
        "SHAP explanation",
        timeout=60,
        json=patient_data,
    )


def validate_patient_data(patient_data):
    """Validate required text and numeric fields before making API calls."""
    required_text = (
        "gender", "race", "admission_type", "admission_source",
        "discharge_disposition", "hba1c_result", "glucose_result",
        "insulin_status", "diabetes_medication",
        "primary_diagnosis_category", "secondary_diagnosis_category",
        "tertiary_diagnosis_category",
    )
    missing = [
        key for key in required_text
        if not isinstance(patient_data.get(key), str)
        or not patient_data[key].strip()
    ]
    if missing:
        return (
            "Complete all required text fields before requesting a "
            "prediction."
        )

    try:
        age = float(patient_data["age_midpoint"])
        if not 0 <= age <= 120:
            raise ValueError
        numeric_fields = (
            "length_of_stay",
            "previous_inpatient_visits",
            "previous_outpatient_visits",
            "previous_emergency_visits",
            "diagnosis_count",
            "lab_procedure_count",
            "procedure_count",
            "medication_count",
            "total_previous_visits",
            "has_previous_inpatient_visit",
        )
        for key in numeric_fields:
            value = patient_data[key]
            if type(value) is not int or value < 0:
                raise ValueError
        if patient_data["has_previous_inpatient_visit"] not in (0, 1):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return (
            "One or more numeric inputs are outside the supported range. "
            "Check age, visit counts, and procedure/diagnosis/medication "
            "counts."
        )
    return None
