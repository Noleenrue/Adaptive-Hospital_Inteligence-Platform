"""HbA1c / glucose "not measured" must be encoded identically in training
and serving, without silently becoming a measured value."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.predictor import ReadmissionPredictor
from src.preprocessing.lab_encoding import (
    INPUT_MAPPING_ID,
    LAB_COLUMNS,
    NOT_MEASURED,
    normalize_lab_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
SAMPLE = PROJECT_ROOT / "scripts" / "_sample_patient.json"
CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates" / "corrected-v3"


def _sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


# ---- normalisation function ------------------------------------------------

@pytest.mark.parametrize(
    "raw", [None, np.nan, "", "None", "none", " None ", "NaN", "Not measured"]
)
def test_unmeasured_values_map_to_explicit_category(raw):
    df = pd.DataFrame({"hba1c_result": [raw], "glucose_result": [raw]})
    out = normalize_lab_inputs(df)

    assert out["hba1c_result"].iloc[0] == NOT_MEASURED
    assert out["glucose_result"].iloc[0] == NOT_MEASURED


@pytest.mark.parametrize(
    "column,value",
    [("hba1c_result", "Norm"), ("hba1c_result", ">7"),
     ("hba1c_result", ">8"), ("glucose_result", "Norm"),
     ("glucose_result", ">200"), ("glucose_result", ">300")],
)
def test_measured_values_are_never_changed(column, value):
    df = pd.DataFrame({
        "hba1c_result": ["Norm"], "glucose_result": ["Norm"],
    })
    df[column] = [value]

    assert normalize_lab_inputs(df)[column].iloc[0] == value


def test_normalisation_does_not_mutate_input_or_other_columns():
    df = pd.DataFrame({
        "hba1c_result": ["None"], "glucose_result": [None],
        "gender": ["Female"], "age_midpoint": [65.0],
    })
    snapshot = df.copy(deep=True)
    out = normalize_lab_inputs(df)

    pd.testing.assert_frame_equal(df, snapshot)
    assert out["gender"].iloc[0] == "Female"
    assert out["age_midpoint"].iloc[0] == 65.0


def test_all_nan_column_is_handled():
    df = pd.DataFrame({
        "hba1c_result": [np.nan, np.nan], "glucose_result": [np.nan, np.nan]
    })

    assert (normalize_lab_inputs(df)[list(LAB_COLUMNS)] == NOT_MEASURED
            ).all().all()


@pytest.mark.skipif(not CSV.exists(), reason="processed data not available")
def test_training_data_and_dashboard_encode_unmeasured_the_same():
    """Training CSV stores 'not measured' as blank; dashboard sends 'None'."""

    csv = pd.read_csv(CSV, usecols=list(LAB_COLUMNS)).head(2000)
    trained = normalize_lab_inputs(csv)
    served = normalize_lab_inputs(
        pd.DataFrame({c: ["None"] for c in LAB_COLUMNS})
    )

    for column in LAB_COLUMNS:
        blanks = csv[column].isna()
        assert blanks.any()
        assert (trained.loc[blanks, column] == served[column].iloc[0]).all()
        assert set(trained[column]) <= {
            NOT_MEASURED, "Norm", ">7", ">8", ">200", ">300"
        }


# ---- deployed corrected-v3 mapping ----------------------------------------

def _transform(predictor, df):
    out = predictor.model.named_steps["preprocessor"].transform(df)
    return out.toarray()[0] if hasattr(out, "toarray") else np.asarray(out)[0]


def test_deployed_model_blank_and_none_inputs_are_equivalent():
    """The deployed v3 manifest opts both spellings into the same mapping."""
    predictor = ReadmissionPredictor()
    blank = pd.DataFrame([{**_sample(), "hba1c_result": np.nan,
                           "glucose_result": np.nan}])
    none = pd.DataFrame([{**_sample(), "hba1c_result": "None",
                          "glucose_result": "None"}])

    assert predictor.model_version == "corrected-v3"
    assert predictor.input_mapping == INPUT_MAPPING_ID
    assert np.array_equal(
        _transform(predictor, predictor.prepare_features(blank)),
        _transform(predictor, predictor.prepare_features(none)),
    )


def test_deployed_model_applies_manifest_declared_mapping():
    predictor = ReadmissionPredictor()
    frame = pd.DataFrame([_sample()])

    assert predictor.model_version == "corrected-v3"
    assert predictor.input_mapping == INPUT_MAPPING_ID
    pd.testing.assert_frame_equal(
        predictor.prepare_features(frame), normalize_lab_inputs(frame)
    )


# ---- candidate with the corrected mapping ----------------------------------

needs_candidate = pytest.mark.skipif(
    not (CANDIDATE_DIR / "manifest.json").exists(),
    reason="corrected-v3 candidate has not been generated",
)


@needs_candidate
def test_candidate_declares_and_validates_input_mapping():
    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert predictor.input_mapping == INPUT_MAPPING_ID
    assert predictor.model_version == "corrected-v3"

    encoder = predictor.model.named_steps["preprocessor"].named_transformers_[
        "cat"].named_steps["onehot"]
    names = predictor.model.named_steps["preprocessor"].transformers_[1][2]
    categories = dict(zip(names, encoder.categories_))

    for column in LAB_COLUMNS:
        assert NOT_MEASURED in list(categories[column])


@needs_candidate
def test_candidate_blank_and_none_inputs_are_identical():
    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    results = []

    for value in (np.nan, None, "None", "Not measured"):
        frame = pd.DataFrame([{**_sample(), "hba1c_result": value,
                               "glucose_result": value}])
        assert np.array_equal(
            _transform(predictor, predictor.prepare_features(frame)),
            _transform(
                predictor,
                predictor.prepare_features(pd.DataFrame([{
                    **_sample(), "hba1c_result": "None",
                    "glucose_result": "None"}])),
            ),
        )
        results.append(predictor.predict(frame)["raw_probability"])

    assert len(set(results)) == 1


@needs_candidate
def test_candidate_unmeasured_differs_from_every_measured_value():
    """'Not measured' is its own category, not a disguised measurement."""

    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    base = predictor.prepare_features(pd.DataFrame([_sample()]))
    vec_none = _transform(predictor, base)

    for column, values in (("hba1c_result", ["Norm", ">7", ">8"]),
                           ("glucose_result", ["Norm", ">200", ">300"])):
        for value in values:
            other = base.copy()
            other[column] = value

            assert not np.array_equal(vec_none, _transform(predictor, other))


@needs_candidate
def test_candidate_explanations_are_faithful_and_use_mapping():
    from src.explainability.shap_explainer import ReadmissionSHAPExplainer

    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)
    explainer = ReadmissionSHAPExplainer(predictor)
    frame = pd.DataFrame([_sample()])

    table, base_value, margin = explainer.contributions(frame)
    raw = predictor.predict(frame)["raw_probability"]

    assert table["shap_value"].sum() + base_value == pytest.approx(
        margin, abs=1e-4
    )
    assert 1 / (1 + np.exp(-margin)) == pytest.approx(raw, abs=1e-5)
    assert any(
        f.endswith("hba1c_result_Not measured") for f in table["feature"]
    )
