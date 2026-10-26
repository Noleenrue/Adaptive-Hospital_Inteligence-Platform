import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src.models import risk_policy
from src.models.risk_policy import assign_risk_tier
from src.preprocessing.lab_encoding import (
    INPUT_MAPPING_ID,
    LAB_COLUMNS,
    NOT_MEASURED,
    normalize_lab_inputs,
)


# Project root:
# src/models/predictor.py -> project root is two levels above "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR_ENV_VAR = "READMISSION_MODEL_DIR"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "readmission_xgb_pipeline.joblib"
CALIBRATOR_FILENAME = "platt_calibrator.joblib"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_MODEL_NAME = "readmission_xgb"
# Deployed artifacts predate manifests; "1.0" matches existing database rows.
LEGACY_MODEL_VERSION = "1.0"
DB_VERSION_MAX_LENGTH = 20

MODEL_PATH = DEFAULT_MODEL_DIR / MODEL_FILENAME
CALIBRATOR_PATH = DEFAULT_MODEL_DIR / CALIBRATOR_FILENAME


def resolve_model_dir(model_dir=None):
    """Return the artifact directory: argument, then env var, then default."""

    if model_dir is None:
        model_dir = os.environ.get(MODEL_DIR_ENV_VAR) or DEFAULT_MODEL_DIR

    model_dir = Path(model_dir)

    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    return model_dir


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReadmissionPredictor:
    """
    Production prediction service for 30-day hospital readmission risk.

    The service:
    1. Loads the trained XGBoost pipeline.
    2. Generates a raw model probability.
    3. Applies Platt probability calibration.
    4. Assigns a risk tier.
    """

    def __init__(
        self,
        model_path=None,
        calibrator_path=None,
        model_dir=None,
    ):
        """
        By default the model and calibrator are loaded together from one
        directory: ``model_dir`` or the READMISSION_MODEL_DIR environment
        variable, falling back to the deployed ``models/`` directory.
        Explicit file paths cannot be combined with ``model_dir``.
        """

        if model_dir is not None and (
            model_path is not None or calibrator_path is not None
        ):
            raise ValueError(
                "Provide either model_dir or explicit file paths, not both."
            )

        if (model_path is None) != (calibrator_path is None):
            raise ValueError(
                "model_path and calibrator_path must be provided together."
            )

        if model_path is not None:
            self.model_dir = None
            self.model_path = Path(model_path)
            self.calibrator_path = Path(calibrator_path)
        else:
            self.model_dir = resolve_model_dir(model_dir)
            self.model_path = self.model_dir / MODEL_FILENAME
            self.calibrator_path = self.model_dir / CALIBRATOR_FILENAME

        self._validate_artifacts()

        self.model = joblib.load(self.model_path)
        self.calibrator = joblib.load(self.calibrator_path)

        self.manifest = self._load_manifest()
        self._validate_compatibility()

        self.input_mapping = (
            self.manifest.get("input_mapping") if self.manifest else None
        )
        self._validate_input_mapping()

        self.artifact_id = self._compute_artifact_id()
        self.model_name = DEFAULT_MODEL_NAME
        self.model_version = (
            str(self.manifest["version"])
            if self.manifest and self.manifest.get("version")
            else LEGACY_MODEL_VERSION
        )

    def _compute_artifact_id(self):
        """Short identifier of the exact model + calibrator file contents."""

        digest = hashlib.sha256()
        digest.update(_sha256(self.model_path).encode())
        digest.update(_sha256(self.calibrator_path).encode())

        return f"sha256:{digest.hexdigest()[:16]}"

    @property
    def db_model_version(self):
        """Model version shortened to fit the VARCHAR(20) database column."""

        if len(self.model_version) <= DB_VERSION_MAX_LENGTH:
            return self.model_version

        suffix = self.artifact_id.split(":")[-1][:8]
        head = self.model_version[: DB_VERSION_MAX_LENGTH - len(suffix) - 1]

        return f"{head}-{suffix}"

    def version_info(self):
        """Identity of the loaded artifacts, for API responses."""

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_id": self.artifact_id,
        }

    def _load_manifest(self):
        """Load the optional manifest stored next to the artifacts."""

        manifest_path = self.model_path.parent / MANIFEST_FILENAME

        if self.model_path.parent != self.calibrator_path.parent:
            return None

        if not manifest_path.exists():
            return None

        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _validate_compatibility(self):
        """
        Check the loaded artifacts are consistent with each other.

        Always checked: the calibrator takes one raw score. When a manifest
        is present (candidate bundles): file hashes, feature schema and risk
        policy must match. Deployed artifacts have no manifest.
        """

        if getattr(self.calibrator, "n_features_in_", 1) != 1:
            raise ValueError(
                "Calibrator must accept exactly one input (the raw score)."
            )

        if self.manifest is None:
            return

        for name, expected in self.manifest.get("files", {}).items():
            artifact = self.model_path.parent / name

            if not artifact.exists() or _sha256(artifact) != expected:
                raise ValueError(
                    f"Artifact {name} does not match its manifest hash."
                )

        expected_features = self.manifest.get("features")
        model_features = list(
            getattr(self.model, "feature_names_in_", [])
        )

        if expected_features is not None and (
            list(expected_features) != model_features
        ):
            raise ValueError(
                "Manifest feature schema does not match the model."
            )

        policy = self.manifest.get("risk_policy")

        if policy is not None:
            active = {
                "low_risk_threshold": risk_policy.LOW_RISK_THRESHOLD,
                "high_risk_threshold": risk_policy.HIGH_RISK_THRESHOLD,
                "intervention_threshold": (
                    risk_policy.INTERVENTION_THRESHOLD
                ),
            }

            if policy != active:
                raise ValueError(
                    "Manifest risk policy does not match the active "
                    f"policy: {policy} != {active}"
                )

    def _validate_input_mapping(self):
        """
        Check the manifest's input mapping is known and that the fitted
        encoder actually contains the category it maps to.

        Artifacts without an ``input_mapping`` entry keep the legacy
        behaviour (inputs are passed to the pipeline unchanged).
        """

        if self.input_mapping is None:
            return

        if self.input_mapping != INPUT_MAPPING_ID:
            raise ValueError(
                f"Unsupported input mapping: {self.input_mapping}"
            )

        preprocessor = self.model.named_steps["preprocessor"]
        columns = dict(
            (name, cols) for name, _, cols in preprocessor.transformers_
        )
        encoder = preprocessor.named_transformers_["cat"].named_steps[
            "onehot"
        ]
        categories = dict(zip(columns["cat"], encoder.categories_))

        for column in LAB_COLUMNS:
            if NOT_MEASURED not in list(categories.get(column, [])):
                raise ValueError(
                    f"Model was not trained with a '{NOT_MEASURED}' "
                    f"category for {column}."
                )

    def prepare_features(self, patient_features):
        """
        Apply the artifact's input mapping, if any.

        Used by both predictions and explanations so they see identical
        model inputs.
        """

        if isinstance(patient_features, dict):
            patient_features = pd.DataFrame([patient_features])

        if self.input_mapping == INPUT_MAPPING_ID:
            return normalize_lab_inputs(patient_features)

        return patient_features

    def _validate_artifacts(self):
        """Ensure required model artifacts exist."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Prediction model not found: {self.model_path}"
            )

        if not self.calibrator_path.exists():
            raise FileNotFoundError(
                f"Probability calibrator not found: {self.calibrator_path}"
            )

    def predict(self, patient_features):
        """
        Predict calibrated 30-day readmission risk.

        Parameters
        ----------
        patient_features : dict or pandas.DataFrame
            Patient features expected by the trained model pipeline.

        Returns
        -------
        dict
            Raw model score, calibrated probability, and risk tier.
        """

        if isinstance(patient_features, dict):
            patient_features = pd.DataFrame([patient_features])

        if not isinstance(patient_features, pd.DataFrame):
            raise TypeError(
                "patient_features must be a dictionary or pandas DataFrame."
            )

        if patient_features.empty:
            raise ValueError("patient_features cannot be empty.")

        # Step 0: identical input mapping to training (if the artifact has one)
        patient_features = self.prepare_features(patient_features)

        # Step 1: XGBoost raw probability
        raw_probability = float(
            self.model.predict_proba(patient_features)[:, 1][0]
        )

        # Step 2: Platt calibration
        calibrated_probability = float(
            self.calibrator.predict_proba(
                [[raw_probability]]
            )[:, 1][0]
        )

        # Step 3: Clinical risk policy
        risk_tier = assign_risk_tier(calibrated_probability)

        return {
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "risk_tier": risk_tier,
        }
