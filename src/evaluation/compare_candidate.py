"""Read-only comparison of the deployed model and a candidate bundle.

Both bundles are scored on the same untouched, patient-level test partition
(the split recorded in the manifests). Nothing is trained, tuned or written
outside reports/. Also measures the effect of the HbA1c/glucose serving
mismatch: scoring blanks as in training vs the literal "None" sent by the
dashboard.

Usage: python -m src.evaluation.compare_candidate [--candidate corrected-v3]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_model import (  # noqa: E402
    DATA_PATH,
    GROUP,
    SEED,
    TARGET,
    expected_calibration_error,
    patient_split,
)
from src.explainability.shap_explainer import ReadmissionSHAPExplainer  # noqa: E402
from src.models import risk_policy  # noqa: E402
from src.models.predictor import ReadmissionPredictor  # noqa: E402
from src.preprocessing.lab_encoding import LAB_COLUMNS  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "candidates"


def score(predictor, X):
    raw = predictor.model.predict_proba(predictor.prepare_features(X))[:, 1]
    return predictor.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def tiers(p):
    return np.where(
        p < risk_policy.LOW_RISK_THRESHOLD, "Low",
        np.where(p < risk_policy.HIGH_RISK_THRESHOLD, "Moderate", "High"),
    )


def metrics(y, p):
    t = tiers(p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece_10_equal_width": float(expected_calibration_error(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "tiers": {
            k: {"n": int((t == k).sum()),
                "observed": float(y[t == k].mean()),
                "mean_predicted": float(p[t == k].mean())}
            for k in ("Low", "Moderate", "High")
        },
    }


def threshold_metrics(y, p, threshold):
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall))
        if precision + recall else 0.0,
        "flagged_rate": float(predicted.mean()),
    }


def tier_transitions(y, p_v2, p_v3):
    old_tier, new_tier = tiers(p_v2), tiers(p_v3)
    rows = []
    matrix = {}

    for old in ("Low", "Moderate", "High"):
        matrix[old] = {}
        for new in ("Low", "Moderate", "High"):
            selected = (old_tier == old) & (new_tier == new)
            count = int(selected.sum())
            observed = float(y[selected].mean()) if count else None
            rows.append({
                "v2_tier": old,
                "v3_tier": new,
                "encounters": count,
                "observed_readmission_rate": observed,
            })
            matrix[old][new] = {
                "encounters": count,
                "observed_readmission_rate": observed,
            }

    return matrix, pd.DataFrame(rows)


def subgroup_metrics(X, y, p_v2, p_v3, threshold):
    sub = X.copy().reset_index(drop=True)
    sub["age_band"] = pd.cut(
        sub["age_midpoint"], [0, 40, 60, 80, 121], right=False,
        labels=["<40", "40-59", "60-79", "80+"],
    ).astype(str)
    rows = []

    for column in ("gender", "race", "age_band", "admission_type"):
        for value, positions in sub.groupby(column, observed=True).groups.items():
            idx = np.asarray(list(positions), dtype=int)
            yy = y[idx]
            for version, pp in (("v2", p_v2[idx]), ("v3", p_v3[idx])):
                positive = int(yy.sum())
                selected = len(idx) >= 200 and 0 < positive < len(idx)
                rows.append({
                    "attribute": column,
                    "group": str(value),
                    "model": version,
                    "encounters": len(idx),
                    "observed_rate": float(yy.mean()),
                    "mean_predicted": float(pp.mean()),
                    "calibration_gap": float(pp.mean() - yy.mean()),
                    "roc_auc": (
                        float(roc_auc_score(yy, pp)) if selected else None
                    ),
                    "flagged_rate": float((pp >= threshold).mean()),
                    "recall_at_threshold": (
                        float(((pp >= threshold) & (yy == 1)).sum() / positive)
                        if positive else None
                    ),
                    "note": (
                        "" if selected
                        else "AUC pending: fewer than 200 encounters or one class only"
                    ),
                })

    return pd.DataFrame(rows)


def serving_skew(predictor, X):
    """Blank (as trained) vs literal "None" (as sent by the dashboard)."""

    blank, served = X.copy(), X.copy()

    for column in LAB_COLUMNS:
        unmeasured = X[column].isna()
        blank.loc[unmeasured, column] = np.nan
        served.loc[unmeasured, column] = "None"

    affected = X[list(LAB_COLUMNS)].isna().any(axis=1).to_numpy()
    p_blank, p_served = score(predictor, blank), score(predictor, served)
    diff = np.abs(p_blank - p_served)

    return {
        "encounters_with_unmeasured_lab": int(affected.sum()),
        "mean_abs_probability_diff": float(diff.mean()),
        "max_abs_probability_diff": float(diff.max()),
        "share_changing_risk_tier": float((tiers(p_blank) != tiers(p_served)).mean()),
    }


def shap_fidelity(predictor, X, n=300):
    explainer = ReadmissionSHAPExplainer(predictor)
    sample = X.sample(n, random_state=SEED)
    errors, lab_share = [], []
    total = {}

    for i in range(len(sample)):
        table, base, margin = explainer.contributions(sample.iloc[[i]])
        errors.append(abs(table["shap_value"].sum() + base - margin))
        absolute = table.set_index("feature")["shap_value"].abs()

        for column in LAB_COLUMNS:
            total[column] = total.get(column, 0.0) + float(
                absolute[[f for f in absolute.index if f"__{column}_" in f]].sum()
            )

        total["all"] = total.get("all", 0.0) + float(absolute.sum())

    return {
        "rows": n,
        "max_reconstruction_error": float(max(errors)),
        "lab_share_of_total_abs_shap": {
            c: total[c] / total["all"] for c in LAB_COLUMNS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="corrected-v3")
    args = parser.parse_args()

    os.environ.pop("READMISSION_MODEL_DIR", None)
    deployed = ReadmissionPredictor()
    candidate = ReadmissionPredictor(model_dir=CANDIDATES_DIR / args.candidate)

    assert list(deployed.model.feature_names_in_) == list(
        candidate.model.feature_names_in_), "feature schemas differ"
    assert deployed.manifest["risk_policy"] == candidate.manifest["risk_policy"]

    df = pd.read_csv(DATA_PATH)
    df = df.loc[~df["discharge_disposition"].astype(str).str.contains(
        "Expired", case=False, na=False)].copy()
    features = list(deployed.model.feature_names_in_)
    X, y, g = df[features], df[TARGET].astype(int), df[GROUP]
    dev, te = patient_split(X, y, g)
    X_test, y_test, g_test = X.iloc[te], y.iloc[te].to_numpy(), g.iloc[te]

    for manifest in (deployed.manifest, candidate.manifest):
        assert len(X_test) == manifest["split"]["test_encounters"]
        assert g_test.nunique() == manifest["split"]["test_patients"]

    assert not set(g_test) & set(g.iloc[dev]), "patient leakage into test"

    p_old, p_new = score(deployed, X_test), score(candidate, X_test)
    transition_matrix, transition_rows = tier_transitions(
        y_test, p_old, p_new
    )
    threshold = risk_policy.INTERVENTION_THRESHOLD
    subgroup_frame = subgroup_metrics(
        X_test, y_test, p_old, p_new, threshold
    )
    subgroup_frame.to_csv(
        REPORTS_DIR / f"candidate_{args.candidate}_subgroups.csv",
        index=False,
    )
    transition_rows.to_csv(
        REPORTS_DIR / f"candidate_{args.candidate}_tier_transitions.csv",
        index=False,
    )
    result = {
        "test_encounters": int(len(X_test)),
        "test_patients": int(g_test.nunique()),
        "thresholds": deployed.manifest["risk_policy"],
        "deployed": {"model_version": deployed.model_version,
                     "artifact_id": deployed.artifact_id,
                     "metrics": metrics(y_test, p_old),
                     "serving_skew": serving_skew(deployed, X_test),
                     "shap": shap_fidelity(deployed, X_test)},
        "candidate": {"model_version": candidate.model_version,
                      "artifact_id": candidate.artifact_id,
                      "input_mapping": candidate.input_mapping,
                      "metrics": metrics(y_test, p_new),
                      "serving_skew": serving_skew(candidate, X_test),
                      "shap": shap_fidelity(candidate, X_test)},
        "candidate_vs_deployed": {
            "mean_abs_probability_diff": float(np.abs(p_new - p_old).mean()),
            "max_abs_probability_diff": float(np.abs(p_new - p_old).max()),
            "share_changing_risk_tier": float((tiers(p_new) != tiers(p_old)).mean()),
            "tier_transition_matrix_v2_to_v3": transition_matrix,
            "threshold_metrics_at_0_15": {
                "v2": threshold_metrics(y_test, p_old, threshold),
                "v3": threshold_metrics(y_test, p_new, threshold),
            },
        },
        "subgroup_comparison_csv": (
            f"candidate_{args.candidate}_subgroups.csv"
        ),
        "tier_transition_csv": (
            f"candidate_{args.candidate}_tier_transitions.csv"
        ),
    }

    out = REPORTS_DIR / f"candidate_{args.candidate}_comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
