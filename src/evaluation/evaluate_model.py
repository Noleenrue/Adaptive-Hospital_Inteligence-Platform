"""Read-only evaluation of the deployed readmission model.

Loads the deployed artifacts (no training, no threshold tuning), rebuilds the
patient-level partitions described in the artifact manifest, and scores the
untouched test partition. All outputs are written to reports/.

Usage: python -m src.evaluation.evaluate_model
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_explainer import ReadmissionSHAPExplainer  # noqa: E402
from src.models import risk_policy  # noqa: E402
from src.models.predictor import ReadmissionPredictor, _sha256  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
TARGET = "readmitted_30_days"
GROUP = "patient_id"
SEED = 42
N_BOOTSTRAP = 500
SHAP_SAMPLE = 2000
PLACEHOLDERS = {"?", "Unknown", "Unknown/Invalid", "Not Available", "Not Mapped", "NULL"}


def expected_calibration_error(y_true, prob, n_bins=10):
    """Same definition as notebook 05: equal-width bins, weighted gap."""

    y_true, prob = np.asarray(y_true), np.asarray(prob)
    ids = np.digitize(prob, np.linspace(0, 1, n_bins + 1)[1:-1], right=True)
    ece = 0.0

    for b in range(n_bins):
        m = ids == b

        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())

    return float(ece)


def patient_split(X, y, groups):
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        .split(X, y, groups=groups)
    )
    return a, b


def core_metrics(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
    }


def threshold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)

    return {
        "threshold": thr, "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tp": int(tp), "precision": precision, "recall": recall,
        "specificity": div(tn, tn + fp), "npv": div(tn, tn + fn),
        "f1": div(2 * precision * recall, precision + recall),
        "accuracy": div(tp + tn, len(y)),
        "flagged_n": int(tp + fp), "flagged_rate": div(tp + fp, len(y)),
        "lift_in_flagged": div(precision, y.mean()),
    }


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    os.environ.pop("READMISSION_MODEL_DIR", None)

    predictor = ReadmissionPredictor()  # deployed models/ directory
    manifest = predictor.manifest

    if manifest is None:
        sys.exit("Deployed artifacts have no manifest; cannot tie them to a split.")

    features = manifest["features"]
    low_t = risk_policy.LOW_RISK_THRESHOLD
    high_t = risk_policy.HIGH_RISK_THRESHOLD
    sel_t = risk_policy.INTERVENTION_THRESHOLD

    # ---- data identity --------------------------------------------------
    data_hash = _sha256(DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)
    expired_mask = df_all["discharge_disposition"].astype(str).str.contains(
        "Expired", case=False, na=False
    )
    excluded_by_category = (
        df_all.loc[expired_mask, "discharge_disposition"].value_counts().to_dict()
    )
    df = df_all.loc[~expired_mask].copy()

    X, y, g = df[features], df[TARGET].astype(int), df[GROUP]
    X = predictor.prepare_features(X)  # identity for artifacts without an input mapping
    dev, te = patient_split(X, y, g)
    Xd, yd, gd = X.iloc[dev], y.iloc[dev], g.iloc[dev]
    tr, ca = patient_split(Xd, yd, gd)
    parts = {
        "train": (Xd.iloc[tr], yd.iloc[tr], gd.iloc[tr]),
        "calibration": (Xd.iloc[ca], yd.iloc[ca], gd.iloc[ca]),
        "test": (X.iloc[te], y.iloc[te], g.iloc[te]),
    }
    X_test, y_test, g_test = parts["test"]
    yt = y_test.to_numpy()

    # ---- integrity checks ----------------------------------------------
    checks = {
        "data file hash equals manifest source hash":
            data_hash == manifest["data"]["source_sha256"],
        "eligible encounters equal manifest":
            len(df) == manifest["data"]["eligible_encounters"],
        "excluded encounters equal manifest":
            int(expired_mask.sum()) == manifest["data"]["excluded_expired_encounters"],
        "split sizes equal manifest":
            (len(parts["train"][0]), len(parts["calibration"][0]), len(X_test))
            == (manifest["split"]["train_encounters"],
                manifest["split"]["calibration_encounters"],
                manifest["split"]["test_encounters"]),
        "patient sets are disjoint (train/cal/test)":
            not (set(parts["train"][2]) & set(parts["calibration"][2]))
            and not (set(parts["train"][2]) & set(g_test))
            and not (set(parts["calibration"][2]) & set(g_test)),
        "no encounter appears in two partitions":
            df.loc[X_test.index, "encounter_id"].isin(
                pd.concat([df.loc[parts["train"][0].index, "encounter_id"],
                           df.loc[parts["calibration"][0].index, "encounter_id"]])
            ).sum() == 0,
        "no Expired discharge in any partition":
            not X["discharge_disposition"].astype(str).str.contains("Expired", case=False).any(),
        "model feature order equals manifest":
            list(predictor.model.feature_names_in_) == list(features),
        "risk thresholds equal manifest policy":
            manifest["risk_policy"] == {
                "low_risk_threshold": low_t, "high_risk_threshold": high_t,
                "intervention_threshold": sel_t,
            },
    }

    # ---- scoring the untouched test partition ---------------------------
    raw = predictor.model.predict_proba(X_test)[:, 1]
    cal = predictor.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    raw_m, cal_m = core_metrics(yt, raw), core_metrics(yt, cal)

    for k in ("roc_auc", "pr_auc", "brier", "log_loss"):
        checks[f"recomputed {k} equals manifest"] = bool(
            np.isclose(cal_m[k], manifest["test_metrics"][k], atol=1e-9)
        )

    # calibration slope / intercept on the logit scale (test set)
    logit = np.log(np.clip(cal, 1e-9, 1 - 1e-9) / (1 - np.clip(cal, 1e-9, 1 - 1e-9)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), yt)
    cal_slope, cal_intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])

    # patient-level (cluster) bootstrap confidence intervals
    rng = np.random.default_rng(SEED)
    by_patient = pd.Series(np.arange(len(yt))).groupby(g_test.to_numpy()).apply(np.array)
    pats = by_patient.to_numpy()
    boot = {k: [] for k in ("roc_auc", "pr_auc", "brier", "log_loss", "ece")}

    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate(pats[rng.integers(0, len(pats), len(pats))])
        for k, v in core_metrics(yt[idx], cal[idx]).items():
            boot[k].append(v)

    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    at_sel = threshold_metrics(yt, cal, sel_t)

    # ---- risk tiers -----------------------------------------------------
    tier = np.where(cal < low_t, "Low", np.where(cal < high_t, "Moderate", "High"))
    tiers = pd.DataFrame([
        {
            "risk_tier": t, "encounters": int((tier == t).sum()),
            "share_of_test": float((tier == t).mean()),
            "readmitted": int(yt[tier == t].sum()),
            "observed_rate": float(yt[tier == t].mean()),
            "mean_predicted": float(cal[tier == t].mean()),
            "patients": int(g_test[tier == t].nunique()),
        }
        for t in ("Low", "Moderate", "High")
    ])
    tiers.loc[len(tiers)] = {
        "risk_tier": "All", "encounters": len(yt), "share_of_test": 1.0,
        "readmitted": int(yt.sum()), "observed_rate": float(yt.mean()),
        "mean_predicted": float(cal.mean()), "patients": int(g_test.nunique()),
    }
    tiers["relative_risk_vs_low"] = tiers["observed_rate"] / tiers.loc[0, "observed_rate"]

    # ---- reliability tables ---------------------------------------------
    ids = np.digitize(cal, np.linspace(0, 1, 11)[1:-1], right=True)
    rel = pd.DataFrame([
        {"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": int((ids == b).sum()),
         "mean_predicted": float(cal[ids == b].mean()),
         "observed_rate": float(yt[ids == b].mean())}
        for b in range(10) if (ids == b).any()
    ])
    q = pd.qcut(cal, 10, duplicates="drop")
    rel_q = pd.DataFrame({"p": cal, "y": yt, "bin": q}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    ).reset_index()
    rel_q["bin"] = rel_q["bin"].astype(str)

    # ---- partitions / imbalance ----------------------------------------
    spw = float(manifest["model"]["scale_pos_weight"])
    part_rows = [
        {"partition": n, "encounters": len(a), "patients": int(c.nunique()),
         "readmitted": int(b.sum()), "readmission_rate": float(b.mean())}
        for n, (a, b, c) in parts.items()
    ]
    part_rows.append({
        "partition": "eligible total", "encounters": len(df),
        "patients": int(g.nunique()), "readmitted": int(y.sum()),
        "readmission_rate": float(y.mean()),
    })
    partitions = pd.DataFrame(part_rows)

    # ---- data quality ---------------------------------------------------
    quality = []

    for col in features:
        s = df[col]
        quality.append({
            "feature": col, "dtype": str(s.dtype), "nulls": int(s.isna().sum()),
            "placeholder_values": int(s.astype(str).isin(PLACEHOLDERS).sum()),
            "distinct": int(s.nunique()),
        })

    quality = pd.DataFrame(quality)
    hospice = df["discharge_disposition"].astype(str).str.contains("hospice", case=False)
    patient_enc = df.groupby(GROUP).size()
    dq = {
        "raw_rows": len(df_all), "excluded_death_related": int(expired_mask.sum()),
        "excluded_by_category": {k: int(v) for k, v in excluded_by_category.items()},
        "eligible_rows": len(df), "eligible_patients": int(g.nunique()),
        "patients_with_multiple_encounters": int((patient_enc > 1).sum()),
        "max_encounters_per_patient": int(patient_enc.max()),
        "duplicate_encounter_ids": int(df["encounter_id"].duplicated().sum()),
        "readmission_rate_excluded_expired": float(df_all.loc[expired_mask, TARGET].mean()),
        "hospice_discharges_still_eligible": int(hospice.sum()),
        "hospice_readmission_rate": float(df.loc[hospice, TARGET].mean()),
        "non_hospice_readmission_rate": float(df.loc[~hospice, TARGET].mean()),
        "hospice_in_test": int(hospice.loc[X_test.index].sum()),
        "class_imbalance_ratio_neg_to_pos_train": float(
            (parts["train"][1] == 0).sum() / (parts["train"][1] == 1).sum()
        ),
        "scale_pos_weight": spw,
    }

    # ---- subgroup performance ------------------------------------------
    sub = X_test.copy()
    sub["age_band"] = pd.cut(
        sub["age_midpoint"], [0, 40, 60, 80, 121], right=False,
        labels=["<40", "40-59", "60-79", "80+"],
    ).astype(str)
    sub_rows = []

    for col in ("gender", "race", "age_band", "admission_type"):
        for val, idx in sub.reset_index(drop=True).groupby(col).groups.items():
            idx = np.asarray(list(idx))
            yy, pp = yt[idx], cal[idx]
            ok = len(idx) >= 200 and 0 < yy.sum() < len(yy)
            sub_rows.append({
                "attribute": col, "group": val, "encounters": len(idx),
                "readmitted": int(yy.sum()), "observed_rate": float(yy.mean()),
                "mean_predicted": float(pp.mean()),
                "calibration_gap": float(pp.mean() - yy.mean()),
                "roc_auc": float(roc_auc_score(yy, pp)) if ok else None,
                "flagged_rate": float((pp >= sel_t).mean()),
                "recall_at_threshold": float(((pp >= sel_t) & (yy == 1)).sum() / yy.sum()) if yy.sum() else None,
                "note": "" if ok else "AUC pending: fewer than 200 encounters or one class only",
            })

    subgroups = pd.DataFrame(sub_rows)

    # ---- explainability (corrected deployed model) ----------------------
    explainer = ReadmissionSHAPExplainer(predictor=predictor)
    checks["SHAP explainer uses the deployed classifier object"] = (
        explainer.classifier is predictor.model.named_steps["classifier"]
    )
    checks["SHAP explainer uses the deployed preprocessor object"] = (
        explainer.preprocessor is predictor.model.named_steps["preprocessor"]
    )
    sample = X_test.sample(SHAP_SAMPLE, random_state=SEED)
    Zs = explainer.preprocessor.transform(sample)  # sparse, as seen by the model
    Zd = Zs.toarray() if hasattr(Zs, "toarray") else np.asarray(Zs)
    names = explainer.preprocessor.get_feature_names_out()
    booster = explainer.classifier.get_booster()
    pipe_margin = explainer.classifier.predict(Zs, output_margin=True)

    # Faithful contributions: XGBoost's native SHAP on the same sparse input
    # the pipeline predicts from (zeros in sparse input are "missing").
    contribs = booster.predict(xgb.DMatrix(Zs, feature_names=None), pred_contribs=True)
    sv, base = contribs[:, :-1], float(contribs[0, -1])
    add_err = float(np.abs(contribs.sum(axis=1) - pipe_margin).max())
    checks["native SHAP contributions sum to the deployed raw margin (max error < 1e-3)"] = add_err < 1e-3
    checks["deployed raw margin maps to deployed raw probability"] = bool(np.allclose(
        1 / (1 + np.exp(-pipe_margin)),
        predictor.model.predict_proba(sample)[:, 1], atol=1e-5,
    ))

    # Fixed /explain path (ReadmissionSHAPExplainer): row-by-row contributions
    # must equal the batch native SHAP and reconstruct the deployed margin.
    check_rows = sample.iloc[:200]
    recon_err, match_err = [], []

    for i in range(len(check_rows)):
        tbl, base_i, margin_i = explainer.contributions(check_rows.iloc[[i]])
        recon_err.append(abs(tbl["shap_value"].sum() + base_i - margin_i))
        match_err.append(abs(tbl["shap_value"].to_numpy() - sv[i]).max())

    checks["explainer contributions reconstruct the deployed margin (200 rows, < 1e-3)"] = max(recon_err) < 1e-3
    checks["explainer contributions equal batch native SHAP (200 rows, < 1e-4)"] = max(match_err) < 1e-4
    explainer_gap = {
        "rows_checked": len(check_rows),
        "max_reconstruction_error": float(max(recon_err)),
        "max_difference_vs_batch_native_shap": float(max(match_err)),
        "note": (
            "Before the fix the explainer densified the input; see "
            "reports/shap_and_lab_encoding_fix.md for the pre-fix gap."
        ),
    }
    cat_cols = list(predictor.model.named_steps["preprocessor"].transformers_[1][2])
    shap_rows = []

    for j, nme in enumerate(names):
        body = nme.split("__", 1)[1]
        orig = next((c for c in sorted(cat_cols, key=len, reverse=True)
                     if nme.startswith("cat__") and body.startswith(c + "_")), body)
        shap_rows.append({"transformed_feature": nme, "original_feature": orig,
                          "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
                          "mean_shap": float(sv[:, j].mean())})

    shap_df = pd.DataFrame(shap_rows).sort_values("mean_abs_shap", ascending=False)
    shap_grouped = (shap_df.groupby("original_feature")["mean_abs_shap"].sum()
                    .sort_values(ascending=False).reset_index())
    shap_grouped["share"] = shap_grouped["mean_abs_shap"] / shap_grouped["mean_abs_shap"].sum()

    # ---- artifacts / identity ------------------------------------------
    pre = predictor.model.named_steps["preprocessor"]
    clf = predictor.model.named_steps["classifier"]
    identity = {
        "model_version": predictor.model_version,
        "artifact_id": predictor.artifact_id,
        "model_file": str(predictor.model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": _sha256(predictor.model_path),
        "calibrator_file": str(predictor.calibrator_path.relative_to(PROJECT_ROOT)),
        "calibrator_sha256": _sha256(predictor.calibrator_path),
        "manifest_sha256": _sha256(predictor.model_path.parent / "manifest.json"),
        "manifest_created_utc": manifest["created_utc"],
        "pipeline_steps": [s for s, _ in predictor.model.steps],
        "numeric_columns": len(pre.transformers_[0][2]),
        "categorical_columns": len(pre.transformers_[1][2]),
        "transformed_features": int(len(names)),
        "classifier": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in clf.get_params().items()
                       if k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                                "colsample_bytree", "scale_pos_weight", "objective", "random_state")},
        "calibrator": {"method": "Platt (LogisticRegression)",
                       "coef": float(predictor.calibrator.coef_.ravel()[0]),
                       "intercept": float(predictor.calibrator.intercept_.ravel()[0])},
        "risk_policy": {"low": low_t, "high": high_t, "intervention": sel_t},
        "data_file": manifest["data"]["source"], "data_sha256": data_hash,
        "environment": manifest["environment"],
    }

    result = {
        "identity": identity, "integrity_checks": {k: bool(v) for k, v in checks.items()},
        "test_calibrated": cal_m, "test_raw": raw_m, "bootstrap_95ci_calibrated": ci,
        "bootstrap": {"method": "patient-level cluster bootstrap", "resamples": N_BOOTSTRAP, "seed": SEED},
        "calibration_slope": cal_slope, "calibration_intercept": cal_intercept,
        "at_selected_threshold": at_sel, "data_quality": dq,
        "shap": {"sample_size": SHAP_SAMPLE, "method": "xgboost native pred_contribs on sparse pipeline input",
                 "base_value_raw_margin": base, "max_additivity_error": add_err,
                 "explain_endpoint_fidelity": explainer_gap},
    }
    (REPORTS_DIR / "evaluation_metrics.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )

    for name, table in {
        "evaluation_risk_tiers.csv": tiers, "evaluation_calibration_bins.csv": rel,
        "evaluation_calibration_deciles.csv": rel_q, "evaluation_partitions.csv": partitions,
        "evaluation_feature_quality.csv": quality, "evaluation_subgroups.csv": subgroups,
        "evaluation_shap_features.csv": shap_df.head(30),
        "evaluation_shap_by_original_feature.csv": shap_grouped,
        "evaluation_confusion_matrix.csv": pd.DataFrame(
            [[at_sel["tn"], at_sel["fp"]], [at_sel["fn"], at_sel["tp"]]],
            index=["actual_no_readmission", "actual_readmitted"],
            columns=["predicted_below_threshold", "predicted_at_or_above_threshold"]),
    }.items():
        table.to_csv(REPORTS_DIR / name, index=name.endswith("matrix.csv"))

    # ---- figures --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 0.5], [0, 0.5], "--", color="grey", label="Perfect calibration")
    ax.plot(rel_q["mean_predicted"], rel_q["observed_rate"], "o-", label="Test deciles")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed readmission rate")
    ax.set_title("Reliability diagram (test set, deployed model)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_reliability_diagram.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    t3 = tiers[tiers["risk_tier"] != "All"]; x = np.arange(3)
    ax.bar(x - .18, t3["observed_rate"], .36, label="Observed")
    ax.bar(x + .18, t3["mean_predicted"], .36, label="Mean predicted")
    ax.set_xticks(x); ax.set_xticklabels(t3["risk_tier"]); ax.set_ylabel("30-day readmission rate")
    ax.set_title("Risk tiers (test set)"); ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figure_risk_tiers.png", dpi=150); plt.close(fig)

    top = shap_grouped.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["original_feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP| (XGBoost raw margin / log-odds scale)")
    ax.set_title("Global SHAP importance (2,000 test encounters)")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "figure_shap_importance.png", dpi=150); plt.close(fig)

    print(json.dumps({"checks": result["integrity_checks"]}, indent=1))
    print("all integrity checks passed:", all(checks.values()))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
