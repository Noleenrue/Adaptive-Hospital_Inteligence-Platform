import pandas as pd
import pytest

from src.models.predictor import ReadmissionPredictor


@pytest.fixture
def predictor():
    """Create the predictor once for the tests."""
    return ReadmissionPredictor()


def test_predictor_loads(predictor):
    """The trained model and calibrator should load successfully."""

    assert predictor.model is not None
    assert predictor.calibrator is not None


def test_prediction_output_structure(predictor):
    """
    The predictor should return the expected prediction fields.
    """

    # The saved pipeline expects the same feature structure
    # used during model training.
    expected_features = list(
        predictor.model.feature_names_in_
    )

    # Create a simple test patient.
    patient = {
        feature: 0 for feature in expected_features
    }

    patient_df = pd.DataFrame([patient])

    result = predictor.predict(patient_df)

    assert isinstance(result, dict)

    assert "raw_probability" in result
    assert "calibrated_probability" in result
    assert "risk_tier" in result


def test_probabilities_are_valid(predictor):
    """
    Raw and calibrated probabilities must remain between 0 and 1.
    """

    expected_features = list(
        predictor.model.feature_names_in_
    )

    patient = {
        feature: 0 for feature in expected_features
    }

    result = predictor.predict(
        pd.DataFrame([patient])
    )

    assert 0.0 <= result["raw_probability"] <= 1.0

    assert (
        0.0
        <= result["calibrated_probability"]
        <= 1.0
    )


def test_risk_tier_is_valid(predictor):
    """Prediction must produce a recognised risk tier."""

    expected_features = list(
        predictor.model.feature_names_in_
    )

    patient = {
        feature: 0 for feature in expected_features
    }

    result = predictor.predict(
        pd.DataFrame([patient])
    )

    assert result["risk_tier"] in {
        "Low",
        "Moderate",
        "High",
    }


def test_empty_dataframe_rejected(predictor):
    """An empty patient record should not be accepted."""

    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame())
