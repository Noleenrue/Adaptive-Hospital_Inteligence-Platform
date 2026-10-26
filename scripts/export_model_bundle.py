"""Export a corrected model, calibrator and risk policy as a candidate bundle.

Reproduces the pipeline of notebooks 05/06 and writes to
models/candidates/<version>/ only. It never touches the production
artifacts in models/.

Usage: python scripts/export_model_bundle.py [--version VERSION]
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.risk_policy import (  # noqa: E402
    HIGH_RISK_THRESHOLD,
    INTERVENTION_THRESHOLD,
    LOW_RISK_THRESHOLD,
)
from src.preprocessing.lab_encoding import (  # noqa: E402
    INPUT_MAPPING_ID,
    LAB_COLUMNS,
    NOT_MEASURED,
    normalize_lab_inputs,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "candidates"
TARGET = "readmitted_30_days"
GROUP = "patient_id"
SEED = 42

FEATURES = [
    "age_midpoint", "gender", "race", "admission_type", "admission_source",
    "discharge_disposition", "length_of_stay", "previous_inpatient_visits",
    "previous_outpatient_visits", "previous_emergency_visits",
    "diagnosis_count", "lab_procedure_count", "procedure_count",
    "medication_count", "hba1c_result", "glucose_result", "insulin_status",
    "diabetes_medication", "primary_diagnosis_category",
    "secondary_diagnosis_category", "tertiary_diagnosis_category",
    "total_previous_visits", "has_previous_inpatient_visit",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patient_split(X, y, groups):
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        .split(X, y, groups=groups)
    )
    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument(
        "--lab-not-measured",
        action="store_true",
        help=(
            "Encode unmeasured HbA1c/glucose as an explicit 'Not measured' "
            "category (same mapping as serving) instead of imputing them."
        ),
    )
    args = parser.parse_args()

    out_dir = CANDIDATES_DIR / args.version
    if out_dir.exists():
        sys.exit(f"Refusing to overwrite existing candidate: {out_dir}")

    df = pd.read_csv(DATA_PATH)
    rows_before = len(df)
    lab_blank_counts = {c: int(df[c].isna().sum()) for c in LAB_COLUMNS}
    if args.lab_not_measured:
        df = normalize_lab_inputs(df)
    expired = df["discharge_disposition"].astype(str).str.contains(
        "Expired", case=False, na=False
    )
    df = df.loc[~expired].copy()

    X, y, groups = df[FEATURES], df[TARGET].astype(int), df[GROUP]

    dev_idx, test_idx = patient_split(X, y, groups)
    X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test, g_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

    tr_idx, cal_idx = patient_split(X_dev, y_dev, g_dev)
    X_train, y_train, g_train = X_dev.iloc[tr_idx], y_dev.iloc[tr_idx], g_dev.iloc[tr_idx]
    X_cal, y_cal, g_cal = X_dev.iloc[cal_idx], y_dev.iloc[cal_idx], g_dev.iloc[cal_idx]

    # Strict patient separation between all three sets.
    sets = {"train": set(g_train), "calibration": set(g_cal), "test": set(g_test)}
    assert not (sets["train"] & sets["calibration"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["calibration"] & sets["test"])

    categorical = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    # Step names must stay "preprocessor"/"classifier" for the SHAP explainer.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    # Platt calibrator is fitted on the calibration set only.
    cal_raw = pipeline.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(random_state=SEED).fit(cal_raw, y_cal)

    test_raw = pipeline.predict_proba(X_test)[:, 1]
    test_prob = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    y_true = y_test.to_numpy()

    tiers = np.where(
        test_prob < LOW_RISK_THRESHOLD, "Low",
        np.where(test_prob < HIGH_RISK_THRESHOLD, "Moderate", "High"),
    )
    tier_summary = {
        tier: {
            "n": int((tiers == tier).sum()),
            "share": float((tiers == tier).mean()),
            "observed_rate": float(y_true[tiers == tier].mean()),
            "mean_predicted": float(test_prob[tiers == tier].mean()),
        }
        for tier in ("Low", "Moderate", "High")
    }

    out_dir.mkdir(parents=True)
    pipeline_path = out_dir / "readmission_xgb_pipeline.joblib"
    calibrator_path = out_dir / "platt_calibrator.joblib"
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(calibrator, calibrator_path)

    manifest = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate - not promoted",
        "files": {
            pipeline_path.name: sha256(pipeline_path),
            calibrator_path.name: sha256(calibrator_path),
        },
        "data": {
            "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256(DATA_PATH),
            "rows_before_filter": rows_before,
            "excluded_expired_encounters": int(expired.sum()),
            "eligible_encounters": len(df),
            "eligible_patients": int(groups.nunique()),
        },
        "split": {
            "method": "GroupShuffleSplit on patient_id, test 20% then calibration 20% of development",
            "random_state": SEED,
            "train_encounters": len(X_train),
            "calibration_encounters": len(X_cal),
            "test_encounters": len(X_test),
            "train_patients": len(sets["train"]),
            "calibration_patients": len(sets["calibration"]),
            "test_patients": len(sets["test"]),
        },
        "features": FEATURES,
        **(
            {"input_mapping": INPUT_MAPPING_ID,
             "input_mapping_detail": {
                 "columns": list(LAB_COLUMNS),
                 "label": NOT_MEASURED,
                 "blank_counts_before_filter": lab_blank_counts,
             }}
            if args.lab_not_measured else {}
        ),
        "model": {
            "scale_pos_weight": scale_pos_weight,
            "n_transformed_features": len(pipeline.named_steps["preprocessor"].get_feature_names_out()),
        },
        "calibrator": {
            "method": "platt",
            "coef": float(calibrator.coef_.ravel()[0]),
            "intercept": float(calibrator.intercept_.ravel()[0]),
        },
        "risk_policy": {
            "low_risk_threshold": LOW_RISK_THRESHOLD,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "intervention_threshold": INTERVENTION_THRESHOLD,
        },
        "test_metrics": {
            "roc_auc": float(roc_auc_score(y_true, test_prob)),
            "pr_auc": float(average_precision_score(y_true, test_prob)),
            "brier": float(brier_score_loss(y_true, test_prob)),
            "log_loss": float(log_loss(y_true, test_prob)),
            "mean_predicted": float(test_prob.mean()),
            "observed_rate": float(y_true.mean()),
            "tiers": tier_summary,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Candidate written to {out_dir}")
    print(json.dumps(manifest["split"], indent=2))
    print(json.dumps(manifest["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
