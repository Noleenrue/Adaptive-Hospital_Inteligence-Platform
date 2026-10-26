"""SHAP explanations must be faithful to the exact deployed pipeline.

The contributions plus the base value must reconstruct the XGBoost raw
margin (log-odds) that the deployed model actually produces, using the same
feature order and missing-value semantics as ``predict``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import ReadmissionSHAPExplainer
from src.models.predictor import ReadmissionPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT_ROOT / "scripts" / "_sample_patient.json"
CSV = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"

MARGIN_TOL = 1e-4


@pytest.fixture(scope="module")
def predictor():
    return ReadmissionPredictor()


@pytest.fixture(scope="module")
def explainer(predictor):
    return ReadmissionSHAPExplainer(predictor)


def _sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def _variants():
    base = _sample()
    yield base
    yield {**base, "previous_inpatient_visits": 0, "total_previous_visits": 0,
           "has_previous_inpatient_visit": 0, "length_of_stay": 1}
    yield {**base, "age_midpoint": 85.0, "previous_inpatient_visits": 6,
           "total_previous_visits": 9, "has_previous_inpatient_visit": 1,
           "discharge_disposition": "Discharged/transferred to SNF"}
    yield {**base, "hba1c_result": ">8", "glucose_result": ">300",
           "insulin_status": "Up"}
    yield {**base, "hba1c_result": "Norm", "glucose_result": "Norm"}


def _logit(p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


@pytest.mark.parametrize("patient", list(_variants()))
def test_contributions_reconstruct_deployed_margin(
    predictor, explainer, patient
):
    frame = pd.DataFrame([patient])
    table, base_value, margin = explainer.contributions(frame)

    reconstructed = float(table["shap_value"].sum() + base_value)
    assert reconstructed == pytest.approx(margin, abs=MARGIN_TOL)

    raw = predictor.predict(frame)["raw_probability"]
    assert margin == pytest.approx(_logit(raw), abs=MARGIN_TOL)


def test_explain_reports_margin_and_probability(predictor, explainer):
    frame = pd.DataFrame([_sample()])
    result = explainer.explain(frame, top_n=5)

    raw = predictor.predict(frame)["raw_probability"]
    assert result["raw_probability"] == pytest.approx(raw, abs=1e-5)
    assert result["margin"] == pytest.approx(_logit(raw), abs=MARGIN_TOL)
    assert result["explanation_scale"] == "XGBoost raw model score"
    assert result["model_version"] == predictor.model_version
    assert result["artifact_id"] == predictor.artifact_id


def test_feature_names_follow_pipeline_order(predictor, explainer):
    frame = pd.DataFrame([_sample()])
    table, _, _ = explainer.contributions(frame)

    expected = list(
        predictor.model.named_steps["preprocessor"].get_feature_names_out()
    )
    assert list(table["feature"]) == expected


def test_feature_values_align_with_transformed_row(predictor, explainer):
    patient = _sample()
    frame = pd.DataFrame([patient])
    table, _, _ = explainer.contributions(frame)

    values = dict(zip(table["feature"], table["feature_value"]))
    prepared = predictor.prepare_features(frame)
    preprocessor = predictor.model.named_steps["preprocessor"]
    row = preprocessor.transform(prepared)
    row = row.toarray()[0] if hasattr(row, "toarray") else np.asarray(row)[0]

    assert np.allclose(table["feature_value"].to_numpy(), row)
    assert values["cat__gender_" + patient["gender"]] == 1.0
    assert values["num__age_midpoint"] == patient["age_midpoint"]
    assert values["num__length_of_stay"] == patient["length_of_stay"]


def test_top_features_sorted_by_absolute_contribution(explainer):
    result = explainer.explain(pd.DataFrame([_sample()]), top_n=8)
    impacts = [abs(f["shap_value"]) for f in result["top_features"]]

    assert len(impacts) == 8
    assert impacts == sorted(impacts, reverse=True)


def test_explain_rejects_multiple_rows(explainer):
    frame = pd.DataFrame([_sample(), _sample()])

    with pytest.raises(ValueError):
        explainer.explain(frame)


def test_dense_conversion_reproduces_original_missing_semantics_bug(
    predictor,
):
    """The old explainer's densification changes the model's raw margin."""

    frame = pd.DataFrame([_sample()])
    preprocessor = predictor.model.named_steps["preprocessor"]
    classifier = predictor.model.named_steps["classifier"]
    sparse_input = preprocessor.transform(frame)
    dense_input = (
        sparse_input.toarray()
        if hasattr(sparse_input, "toarray")
        else np.asarray(sparse_input)
    )

    sparse_margin = float(
        classifier.predict(sparse_input, output_margin=True)[0]
    )
    dense_margin = float(
        classifier.predict(dense_input, output_margin=True)[0]
    )

    assert abs(sparse_margin - dense_margin) > 0.1


@pytest.mark.skipif(not CSV.exists(), reason="processed data not available")
def test_single_row_and_batch_predictions_agree(predictor):
    """Sparse/dense output depends on batch density; scoring must not."""

    frame = pd.read_csv(CSV).sample(150, random_state=7)[
        list(predictor.model.feature_names_in_)
    ]
    frame = predictor.prepare_features(frame)

    batch = predictor.model.predict_proba(frame)[:, 1]
    single = np.array([
        predictor.model.predict_proba(frame.iloc[[i]])[:, 1][0]
        for i in range(len(frame))
    ])

    assert np.allclose(batch, single, atol=1e-6)


@pytest.mark.skipif(not CSV.exists(), reason="processed data not available")
def test_reconstruction_on_real_rows(predictor, explainer):
    frame = pd.read_csv(CSV).sample(40, random_state=11)[
        list(predictor.model.feature_names_in_)
    ]

    for i in range(len(frame)):
        row = frame.iloc[[i]]
        _, base_value, margin = explainer.contributions(row)
        table, _, _ = explainer.contributions(row)
        raw = predictor.predict(row)["raw_probability"]

        assert table["shap_value"].sum() + base_value == pytest.approx(
            margin, abs=MARGIN_TOL
        )
        assert margin == pytest.approx(_logit(raw), abs=MARGIN_TOL)
