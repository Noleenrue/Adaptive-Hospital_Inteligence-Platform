import json
import shutil

import joblib
import pandas as pd
import pytest

from src.models import predictor as predictor_module
from src.models.predictor import (
    CALIBRATOR_FILENAME,
    DEFAULT_MODEL_DIR,
    MANIFEST_FILENAME,
    MODEL_DIR_ENV_VAR,
    MODEL_FILENAME,
    ReadmissionPredictor,
)

CANDIDATE_DIR = (
    DEFAULT_MODEL_DIR / "candidates" / "corrected-v2-candidate"
)

needs_candidate = pytest.mark.skipif(
    not (CANDIDATE_DIR / MANIFEST_FILENAME).exists(),
    reason="candidate bundle not generated",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(MODEL_DIR_ENV_VAR, raising=False)


@pytest.fixture
def candidate_copy(tmp_path):
    """Work on a copy so the real candidate is never modified."""

    target = tmp_path / "bundle"
    shutil.copytree(CANDIDATE_DIR, target)
    return target


def _patient(predictor):
    return pd.DataFrame(
        [{f: 0 for f in predictor.model.feature_names_in_}]
    )


def test_default_loads_deployed_artifacts():
    predictor = ReadmissionPredictor()

    assert predictor.model_dir == DEFAULT_MODEL_DIR
    assert predictor.model_path == DEFAULT_MODEL_DIR / MODEL_FILENAME
    assert (
        predictor.calibrator_path == DEFAULT_MODEL_DIR / CALIBRATOR_FILENAME
    )
    # A manifest exists only once a bundle has been promoted.
    assert (predictor.manifest is None) == (
        not (DEFAULT_MODEL_DIR / MANIFEST_FILENAME).exists()
    )


@needs_candidate
def test_env_var_selects_candidate_directory(monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(CANDIDATE_DIR))

    predictor = ReadmissionPredictor()

    assert predictor.model_path.parent == CANDIDATE_DIR
    assert predictor.calibrator_path.parent == CANDIDATE_DIR
    assert predictor.manifest["version"] == "corrected-v2-candidate"


@needs_candidate
def test_argument_overrides_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path / "missing"))

    predictor = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert predictor.model_path.parent == CANDIDATE_DIR


@needs_candidate
def test_candidate_predicts_valid_output(monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(CANDIDATE_DIR))
    predictor = ReadmissionPredictor()

    result = predictor.predict(_patient(predictor))

    assert 0 <= result["raw_probability"] <= 1
    assert 0 <= result["calibrated_probability"] <= 1
    assert result["risk_tier"] in {"Low", "Moderate", "High"}


@needs_candidate
def test_candidate_schema_matches_deployed_and_manifest():
    deployed = ReadmissionPredictor()
    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert list(candidate.model.feature_names_in_) == list(
        deployed.model.feature_names_in_
    )
    assert candidate.manifest["features"] == list(
        candidate.model.feature_names_in_
    )
    assert list(candidate.model.named_steps) == list(
        deployed.model.named_steps
    )


@needs_candidate
def test_candidate_risk_policy_matches_active_policy():
    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert candidate.manifest["risk_policy"] == {
        "low_risk_threshold": 0.10,
        "high_risk_threshold": 0.15,
        "intervention_threshold": 0.15,
    }


@needs_candidate
def test_candidate_calibrator_belongs_to_candidate_model():
    """Calibrator must be the one fitted for this model, not the deployed one."""

    candidate = ReadmissionPredictor(model_dir=CANDIDATE_DIR)

    assert candidate.calibrator.coef_[0][0] == pytest.approx(
        candidate.manifest["calibrator"]["coef"]
    )


@needs_candidate
def test_tampered_calibrator_is_rejected(candidate_copy):
    # Independent of what is deployed: write a calibrator with other weights.
    calibrator = joblib.load(candidate_copy / CALIBRATOR_FILENAME)
    calibrator.coef_ = calibrator.coef_ * 1.5
    joblib.dump(calibrator, candidate_copy / CALIBRATOR_FILENAME)

    with pytest.raises(ValueError, match="manifest hash"):
        ReadmissionPredictor(model_dir=candidate_copy)


@needs_candidate
def test_feature_schema_mismatch_is_rejected(candidate_copy):
    path = candidate_copy / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["features"] = manifest["features"][:-1]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature schema"):
        ReadmissionPredictor(model_dir=candidate_copy)


@needs_candidate
def test_risk_policy_mismatch_is_rejected(candidate_copy):
    path = candidate_copy / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["risk_policy"]["high_risk_threshold"] = 0.20
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="risk policy"):
        ReadmissionPredictor(model_dir=candidate_copy)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReadmissionPredictor(model_dir=tmp_path / "nope")


def test_missing_calibrator_raises(tmp_path):
    shutil.copy(DEFAULT_MODEL_DIR / MODEL_FILENAME, tmp_path / MODEL_FILENAME)

    with pytest.raises(FileNotFoundError, match="calibrator"):
        ReadmissionPredictor(model_dir=tmp_path)


def test_model_dir_and_explicit_paths_are_exclusive(tmp_path):
    with pytest.raises(ValueError):
        ReadmissionPredictor(
            model_dir=tmp_path,
            model_path=tmp_path / MODEL_FILENAME,
            calibrator_path=tmp_path / CALIBRATOR_FILENAME,
        )


def test_single_explicit_path_is_rejected():
    with pytest.raises(ValueError):
        ReadmissionPredictor(model_path=predictor_module.MODEL_PATH)


def test_explicit_paths_still_supported():
    predictor = ReadmissionPredictor(
        model_path=predictor_module.MODEL_PATH,
        calibrator_path=predictor_module.CALIBRATOR_PATH,
    )

    assert predictor.model is not None
