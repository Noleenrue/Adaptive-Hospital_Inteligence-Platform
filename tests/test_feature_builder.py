import pandas as pd
import pytest

from src.features.feature_builder import (
    MODEL_FEATURES,
    select_model_features,
)


def make_valid_patient():
    return {
        feature: 1
        for feature in MODEL_FEATURES
    }


def test_correct_number_of_features():
    assert len(MODEL_FEATURES) == 23


def test_feature_selection():
    patient = make_valid_patient()

    # Extra field should not reach the ML model
    patient["patient_name"] = "Example Patient"

    df = pd.DataFrame([patient])

    result = select_model_features(df)

    assert list(result.columns) == MODEL_FEATURES
    assert result.shape == (1, 23)
    assert "patient_name" not in result.columns


def test_missing_feature_rejected():
    patient = make_valid_patient()

    del patient["age_midpoint"]

    df = pd.DataFrame([patient])

    with pytest.raises(
        ValueError,
        match="Missing required model features",
    ):
        select_model_features(df)
