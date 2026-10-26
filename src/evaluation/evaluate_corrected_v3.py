"""Read-only, manifest-verified evaluation of the deployed corrected-v3 model.

Run from the project root with ``python -m src.evaluation.evaluate_corrected_v3``.
The evaluator scores only the untouched patient-level test partition and
refuses to overwrite any corrected-v3 output.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import risk_policy  # noqa: E402
from src.models.predictor import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ReadmissionPredictor,
    _sha256,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
EXPECTED_VERSION = "corrected-v3"
EXPECTED_ARTIFACT_ID = "sha256:21438015f642839e"
SEED = 42
N_BOOTSTRAP = 500
TARGET = "readmitted_30_days"
GROUP = "patient_id"
OUTPUT_NAMES = (
    "corrected_v3_model_evaluation_report.md",
    "corrected_v3_evaluation_metrics.json",
    "corrected_v3_evaluation_partitions.csv",
    "corrected_v3_confusion_matrix.csv",
    "corrected_v3_calibration_bins.csv",
    "corrected_v3_calibration_deciles.csv",
    "corrected_v3_reliability_diagram.png",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def patient_split(X, y, groups):
    """Reproduce the exporter's deterministic patient-group split."""
    train_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=SEED
        ).split(X, y, groups=groups)
    )
    return train_idx, test_idx


def reconstruct_partitions(df, manifest):
    """Rebuild and verify the original train/calibration/test row partitions."""
    data_meta = manifest["data"]
    split_meta = manifest["split"]
    features = manifest["features"]

    _require(
        split_meta.get("method")
        == "GroupShuffleSplit on patient_id, test 20% then calibration 20% of development",
        "Split method differs from the corrected-v3 training procedure.",
    )
    _require(
        split_meta.get("random_state") == SEED,
        "Split random seed differs from the corrected-v3 training procedure.",
    )
    _require(
        list(features) == list(
            ["age_midpoint", "gender", "race", "admission_type",
             "admission_source", "discharge_disposition", "length_of_stay",
             "previous_inpatient_visits", "previous_outpatient_visits",
             "previous_emergency_visits", "diagnosis_count",
             "lab_procedure_count", "procedure_count", "medication_count",
             "hba1c_result", "glucose_result", "insulin_status",
             "diabetes_medication", "primary_diagnosis_category",
             "secondary_diagnosis_category", "tertiary_diagnosis_category",
             "total_previous_visits", "has_previous_inpatient_visit"]
        ),
        "Manifest feature schema differs from corrected-v3 schema.",
    )
    _require(
        len(df) == data_meta["rows_before_filter"],
        "Source dataset row count does not match the corrected-v3 manifest.",
    )

    expired = df["discharge_disposition"].astype(str).str.contains(
        "Expired", case=False, na=False
    )
    _require(
        int(expired.sum()) == data_meta["excluded_expired_encounters"],
        "Expired-discharge count does not match the corrected-v3 manifest.",
    )
    eligible = df.loc[~expired].copy()
    _require(
        len(eligible) == data_meta["eligible_encounters"],
        "Eligible encounter count does not match the corrected-v3 manifest.",
    )
    _require(
        int(eligible[GROUP].nunique()) == data_meta["eligible_patients"],
        "Eligible patient count does not match the corrected-v3 manifest.",
    )
    _require(
        not eligible["discharge_disposition"].astype(str).str.contains(
            "Expired", case=False, na=False
        ).any(),
        "Death-related encounters remain after the manifest filtering rule.",
    )

    X, y, groups = (
        eligible[features],
        eligible[TARGET].astype(int),
        eligible[GROUP],
    )
    dev_idx, test_idx = patient_split(X, y, groups)
    X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
    groups_dev = groups.iloc[dev_idx]
    train_rel, cal_rel = patient_split(X_dev, y_dev, groups_dev)

    partitions = {
        "train": (X_dev.iloc[train_rel], y_dev.iloc[train_rel],
                  groups_dev.iloc[train_rel]),
        "calibration": (X_dev.iloc[cal_rel], y_dev.iloc[cal_rel],
                       groups_dev.iloc[cal_rel]),
        "test": (X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]),
    }
    actual_groups = {
        name: set(partition[2])
        for name, partition in partitions.items()
    }
    _require(
        not (actual_groups["train"] & actual_groups["calibration"]
             or actual_groups["train"] & actual_groups["test"]
             or actual_groups["calibration"] & actual_groups["test"]),
        "Reconstructed patient partitions overlap.",
    )
    _require(
        eligible.loc[partitions["test"][0].index, "encounter_id"].isin(
            pd.concat([
                eligible.loc[partitions["train"][0].index, "encounter_id"],
                eligible.loc[partitions["calibration"][0].index,
                             "encounter_id"],
            ])
        ).sum() == 0,
        "An encounter appears in more than one reconstructed partition.",
    )

    for name, (part_X, part_y, part_groups) in partitions.items():
        _require(
            len(part_X) == split_meta[f"{name}_encounters"],
            f"{name} encounter count differs from the manifest.",
        )
        _require(
            int(part_groups.nunique()) == split_meta[f"{name}_patients"],
            f"{name} patient count differs from the manifest.",
        )
        _require(
            set(part_y.unique()) == {0, 1},
            f"{name} partition does not contain both target classes.",
        )

    return eligible, partitions, expired


def expected_calibration_error(y_true, probability, n_bins=10):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    bin_ids = np.digitize(
        probability, np.linspace(0, 1, n_bins + 1)[1:-1], right=True
    )
    return float(sum(
        np.mean(bin_ids == bin_index)
        * abs(
            probability[bin_ids == bin_index].mean()
            - y_true[bin_ids == bin_index].mean()
        )
        for bin_index in range(n_bins)
        if np.any(bin_ids == bin_index)
    ))


def threshold_metrics(y_true, probability, threshold):
    predicted = np.asarray(probability) >= threshold
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted.astype(int), labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(
            y_true, predicted, zero_division=0
        )),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def calibration_slope_intercept(y_true, probability):
    """Diagnostic calibration regression; does not update the calibrator."""
    probability = np.clip(np.asarray(probability), 1e-9, 1 - 1e-9)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    regression = LogisticRegression(C=1e6, max_iter=1000)
    regression.fit(logit, np.asarray(y_true))
    return {
        "slope": float(regression.coef_[0, 0]),
        "intercept": float(regression.intercept_[0]),
    }


def _core_metrics(y_true, probability, threshold):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(
            y_true, probability >= threshold
        )),
        "precision": float(precision_score(
            y_true, probability >= threshold, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, probability >= threshold, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, probability >= threshold, zero_division=0
        )),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "ece_10_equal_width": expected_calibration_error(
            y_true, probability
        ),
    }


def patient_cluster_bootstrap(y_true, probability, groups, threshold,
                              n_resamples=N_BOOTSTRAP, seed=SEED):
    """Percentile CIs from patient-cluster resampling with replacement."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    sampled_metrics = {}
    valid_resamples = 0
    single_class_resamples = 0

    for _ in range(n_resamples):
        _, indices = cluster_resample_indices(groups, rng)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size != 2:
            single_class_resamples += 1
            continue

        values = _core_metrics(
            sampled_y, probability[indices], threshold
        )
        values.update(calibration_slope_intercept(
            sampled_y, probability[indices]
        ))
        for name, value in values.items():
            sampled_metrics.setdefault(name, []).append(value)
        valid_resamples += 1

    _require(valid_resamples > 0, "No valid patient-cluster bootstrap samples.")
    intervals = {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "valid_resamples": len(values),
        }
        for name, values in sampled_metrics.items()
    }
    return {
        "method": "patient-level cluster bootstrap, sampled with replacement",
        "requested_resamples": int(n_resamples),
        "valid_resamples": int(valid_resamples),
        "single_class_resamples_skipped": int(single_class_resamples),
        "seed": int(seed),
        "confidence_level": 0.95,
        "intervals": intervals,
    }


def cluster_resample_indices(groups, rng):
    """Sample patient IDs, retaining every encounter for each sampled ID."""
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    rows_by_group = {
        patient: np.flatnonzero(groups == patient)
        for patient in unique_groups
    }
    sampled_groups = rng.choice(
        unique_groups, size=len(unique_groups), replace=True
    )
    return sampled_groups, np.concatenate(
        [rows_by_group[patient] for patient in sampled_groups]
    )


def calibration_tables(y_true, probability):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    equal_ids = np.digitize(
        probability, np.linspace(0, 1, 11)[1:-1], right=True
    )
    equal_rows = []
    for bin_index in range(10):
        selected = equal_ids == bin_index
        if selected.any():
            equal_rows.append({
                "bin": f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}",
                "encounters": int(selected.sum()),
                "mean_predicted": float(probability[selected].mean()),
                "observed_rate": float(y_true[selected].mean()),
            })
    decile = pd.qcut(probability, 10, duplicates="drop")
    decile_frame = pd.DataFrame({
        "probability": probability,
        "outcome": y_true,
        "bin": decile,
    })
    decile_table = decile_frame.groupby(
        "bin", observed=True
    ).agg(
        encounters=("outcome", "size"),
        mean_predicted=("probability", "mean"),
        observed_rate=("outcome", "mean"),
    ).reset_index()
    decile_table["bin"] = decile_table["bin"].astype(str)
    return pd.DataFrame(equal_rows), decile_table


def _verify_bundle_and_data():
    model_dir = DEFAULT_MODEL_DIR.resolve()
    manifest_path = model_dir / "manifest.json"
    _require(manifest_path.is_file(), f"Missing deployed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("version") == EXPECTED_VERSION
        and manifest.get("status") == "promoted",
        "Deployed manifest is not the promoted corrected-v3 bundle.",
    )
    predictor = ReadmissionPredictor(model_dir=model_dir)
    _require(
        predictor.model_version == EXPECTED_VERSION,
        "Loaded model version differs from corrected-v3.",
    )
    _require(
        predictor.artifact_id == EXPECTED_ARTIFACT_ID,
        "Loaded model artifact ID differs from corrected-v3.",
    )
    _require(
        manifest.get("risk_policy") == {
            "low_risk_threshold": risk_policy.LOW_RISK_THRESHOLD,
            "high_risk_threshold": risk_policy.HIGH_RISK_THRESHOLD,
            "intervention_threshold": risk_policy.INTERVENTION_THRESHOLD,
        },
        "Active risk thresholds differ from the corrected-v3 manifest.",
    )

    data_path = PROJECT_ROOT / manifest["data"]["source"]
    _require(data_path.is_file(), f"Missing evaluation dataset: {data_path}")
    data_hash = _sha256(data_path)
    _require(
        data_hash == manifest["data"]["source_sha256"],
        "Dataset SHA-256 does not match corrected-v3 manifest.",
    )
    return predictor, manifest, data_path, data_hash


def _evaluation_report(metrics, manifest, data_hash, bootstrap):
    threshold = metrics["threshold_metrics"]["threshold"]
    threshold_metrics_result = metrics["threshold_metrics"]
    ci = bootstrap["intervals"]

    def formatted_interval(name, digits=4):
        item = ci.get(name)
        if item is None:
            return "Not available"
        return (
            f"{item['lower_95']:.{digits}f}–{item['upper_95']:.{digits}f} "
            f"(valid {item['valid_resamples']})"
        )

    return f"""# corrected-v3 Model Quality and Reliability Evaluation

**Model:** `{EXPECTED_VERSION}`
**Artifact ID:** `{EXPECTED_ARTIFACT_ID}`
**Evaluation population:** held-out test set, {metrics['test_encounters']:,} encounters from {metrics['test_patients']:,} patients
**Intended prediction time:** **at hospital discharge**
**Clinical status:** Research prototype; **not clinically validated**.

## Evaluation provenance

The deployed model and calibrator were loaded without modification. Their identity, manifest file hashes, dataset SHA-256, row counts, expired-discharge exclusion, deterministic random seed ({SEED}), split sizes, patient counts, and pairwise patient separation were checked against the promoted corrected-v3 manifest before scoring. The data hash was `{data_hash}`.

The dataset contains {manifest['data']['rows_before_filter']:,} rows before filtering. Encounters whose `discharge_disposition` contains “Expired” were excluded before splitting ({manifest['data']['excluded_expired_encounters']:,} excluded; {manifest['data']['eligible_encounters']:,} eligible). GroupShuffleSplit on `patient_id` produced {manifest['split']['train_encounters']:,} train / {manifest['split']['calibration_encounters']:,} calibration / {manifest['split']['test_encounters']:,} test encounters. Patient counts were {manifest['split']['train_patients']:,} / {manifest['split']['calibration_patients']:,} / {manifest['split']['test_patients']:,}. The test partition alone was scored. No predictive model or production calibrator was fit or recalibrated, and no threshold tuning used test outcomes. Calibration slope/intercept below are diagnostic statistics only.

## Test-set performance

Threshold-dependent measures use the existing intervention threshold **{threshold:.2f}**; thresholds were not changed.

| Metric | Test result | Patient-cluster bootstrap 95% CI |
|---|---:|---:|
| ROC-AUC | {metrics['roc_auc']:.4f} | {formatted_interval('roc_auc')} |
| PR-AUC (Average Precision) | {metrics['pr_auc']:.4f} | {formatted_interval('pr_auc')} |
| Accuracy | {threshold_metrics_result['accuracy']:.4f} | {formatted_interval('accuracy')} |
| Precision | {threshold_metrics_result['precision']:.4f} | {formatted_interval('precision')} |
| Recall / sensitivity | {threshold_metrics_result['recall']:.4f} | {formatted_interval('recall')} |
| F1-score | {threshold_metrics_result['f1']:.4f} | {formatted_interval('f1')} |
| Brier score | {metrics['brier']:.4f} | {formatted_interval('brier')} |
| Log loss | {metrics['log_loss']:.4f} | {formatted_interval('log_loss')} |
| ECE (10 equal-width bins) | {metrics['ece_10_equal_width']:.4f} | {formatted_interval('ece_10_equal_width')} |

Confusion matrix at threshold {threshold:.2f} (rows actual [no readmission, readmitted], columns predicted [below threshold, at/above threshold]): **[[{threshold_metrics_result['tn']}, {threshold_metrics_result['fp']}], [{threshold_metrics_result['fn']}, {threshold_metrics_result['tp']}]]**.

The bootstrap sampled patients with replacement and retained all encounters for each sampled patient. It requested {bootstrap['requested_resamples']} resamples, of which {bootstrap['valid_resamples']} contained both outcome classes and were used; {bootstrap['single_class_resamples_skipped']} single-class resamples were skipped. Per-metric valid counts are in the JSON output.

## Calibration

Calibration slope: **{metrics['calibration_slope']:.4f}** (95% CI {formatted_interval('slope')}).
Calibration intercept on the logit scale: **{metrics['calibration_intercept']:.4f}** (95% CI {formatted_interval('intercept')}).
Mean predicted risk: **{metrics['mean_predicted']:.4f}**; observed readmission rate: **{metrics['observed_rate']:.4f}**.

The slope and intercept are diagnostic logistic recalibration regression estimates computed on the held-out predictions and outcomes. They were not applied to predictions and did not modify the Platt calibrator. The reliability diagram and equal-width / quantile-bin tables are saved alongside this report. These are single-split estimates and do not establish calibration in future or different populations.

## Prediction-time availability and limitations

This is a **discharge-time readmission prediction model**, not an admission-time model. Discharge disposition and length of stay are only known at or near discharge. Final diagnoses and encounter-completion details—including current-stay laboratory results, procedure counts, and medication information—may be unavailable or incomplete earlier. Using these inputs for an earlier prediction would create a feature-availability mismatch and could constitute temporal leakage. The intended scoring workflow must enforce the discharge-time contract.

Death-related discharges are excluded because a patient who dies during the encounter is not eligible for a post-discharge readmission prediction. This does not exclude hospice discharges. The model uses weighted XGBoost training for class imbalance and a previously fitted Platt calibrator; neither was refit during this evaluation.

This evaluation uses one patient-disjoint random split of a single dataset. It is not temporal, hospital-held-out, or external validation. Performance may differ across hospitals, time periods, coding systems, and populations. No clinical utility, fairness, or prospective impact is established. The model is **not clinically validated** and should not be used as a standalone clinical decision.

## Versioning note

Existing generic `evaluation_*` outputs and `model_evaluation_report.md` are retained as historical corrected-v2 results; they were not overwritten or used as corrected-v3 results. The v3-versus-v2 comparison file contains both versions and was used only to confirm provenance; metrics above are calculated from the loaded corrected-v3 bundle and its verified test split.
"""


def main():
    predictor, manifest, data_path, data_hash = _verify_bundle_and_data()
    destinations = [REPORTS_DIR / name for name in OUTPUT_NAMES]
    existing = [path for path in destinations if path.exists()]
    _require(
        not existing,
        "Refusing to overwrite corrected-v3 evaluation output(s): "
        + ", ".join(str(path) for path in existing),
    )

    df_all = pd.read_csv(data_path)
    _require(
        _sha256(data_path) == manifest["data"]["source_sha256"],
        "Dataset changed during evaluation preflight.",
    )
    eligible, partitions, expired = reconstruct_partitions(
        df_all, manifest
    )
    X_test, y_test, groups_test = partitions["test"]
    _require(
        list(X_test.columns) == list(predictor.model.feature_names_in_),
        "Reconstructed test feature order differs from the loaded model.",
    )
    X_test = predictor.prepare_features(X_test)
    raw_probability = predictor.model.predict_proba(X_test)[:, 1]
    probability = predictor.calibrator.predict_proba(
        raw_probability.reshape(-1, 1)
    )[:, 1]
    y_true = y_test.to_numpy()
    threshold = risk_policy.INTERVENTION_THRESHOLD

    metrics = _core_metrics(y_true, probability, threshold)
    threshold_result = threshold_metrics(y_true, probability, threshold)
    slope_intercept = calibration_slope_intercept(y_true, probability)
    equal_bins, deciles = calibration_tables(y_true, probability)
    bootstrap = patient_cluster_bootstrap(
        y_true, probability, groups_test.to_numpy(), threshold
    )
    partition_rows = []
    for name, (_, labels, groups) in partitions.items():
        partition_rows.append({
            "partition": name,
            "encounters": int(len(labels)),
            "patients": int(groups.nunique()),
            "readmitted": int(labels.sum()),
            "readmission_rate": float(labels.mean()),
        })
    partition_rows.append({
        "partition": "eligible total",
        "encounters": int(len(eligible)),
        "patients": int(eligible[GROUP].nunique()),
        "readmitted": int(eligible[TARGET].sum()),
        "readmission_rate": float(eligible[TARGET].mean()),
    })
    partitions_table = pd.DataFrame(partition_rows)

    result = {
        "identity": {
            "model_version": predictor.model_version,
            "artifact_id": predictor.artifact_id,
            "model_sha256": _sha256(predictor.model_path),
            "calibrator_sha256": _sha256(predictor.calibrator_path),
            "manifest_sha256": _sha256(
                predictor.model_path.parent / "manifest.json"
            ),
            "status": manifest["status"],
        },
        "data": {
            "path": str(data_path.relative_to(PROJECT_ROOT)),
            "sha256": data_hash,
            "raw_encounters": int(len(df_all)),
            "expired_excluded": int(expired.sum()),
            "eligible_encounters": int(len(eligible)),
            "eligible_patients": int(eligible[GROUP].nunique()),
            "filter_rule": (
                'exclude discharge_disposition containing "Expired" '
                "case-insensitively"
            ),
            "split_method": manifest["split"]["method"],
            "random_state": SEED,
        },
        "test_encounters": int(len(y_true)),
        "test_patients": int(groups_test.nunique()),
        "class_distribution": {
            "not_readmitted": int((y_true == 0).sum()),
            "readmitted": int((y_true == 1).sum()),
            "readmission_rate": float(y_true.mean()),
            "mean_predicted_probability": float(probability.mean()),
        },
        "threshold_metrics": threshold_result,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "log_loss": metrics["log_loss"],
        "ece_10_equal_width": metrics["ece_10_equal_width"],
        "calibration_slope": slope_intercept["slope"],
        "calibration_intercept": slope_intercept["intercept"],
        "bootstrap": bootstrap,
        "calibration_analysis": {
            "method": (
                "diagnostic logistic regression of outcome on logit of "
                "held-out calibrated probabilities"
            ),
            "calibrator_refit": False,
            "reliability_diagram": (
                "corrected_v3_reliability_diagram.png"
            ),
            "equal_width_bins": (
                "corrected_v3_calibration_bins.csv"
            ),
            "quantile_bins": (
                "corrected_v3_calibration_deciles.csv"
            ),
        },
        "prediction_time": "at hospital discharge",
        "clinical_validation_status": "not clinically validated",
    }

    report_metrics = {
        **metrics,
        "threshold_metrics": threshold_result,
        "calibration_slope": slope_intercept["slope"],
        "calibration_intercept": slope_intercept["intercept"],
        "mean_predicted": float(probability.mean()),
        "observed_rate": float(y_true.mean()),
        "test_encounters": int(len(y_true)),
        "test_patients": int(groups_test.nunique()),
    }
    report = _evaluation_report(
        report_metrics, manifest, data_hash, bootstrap
    )
    matrix = pd.DataFrame(
        [[threshold_result["tn"], threshold_result["fp"]],
         [threshold_result["fn"], threshold_result["tp"]]],
        index=["actual_no_readmission", "actual_readmitted"],
        columns=["predicted_below_threshold",
                 "predicted_at_or_above_threshold"],
    )
    REPORTS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".corrected-v3-evaluation-", dir=REPORTS_DIR
    ) as staging:
        stage = Path(staging)
        (stage / OUTPUT_NAMES[0]).write_text(report, encoding="utf-8")
        (stage / OUTPUT_NAMES[1]).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        partitions_table.to_csv(stage / OUTPUT_NAMES[2], index=False)
        matrix.to_csv(stage / OUTPUT_NAMES[3], index=True)
        equal_bins.to_csv(stage / OUTPUT_NAMES[4], index=False)
        deciles.to_csv(stage / OUTPUT_NAMES[5], index=False)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey",
                label="Perfect calibration")
        ax.plot(deciles["mean_predicted"], deciles["observed_rate"],
                "o-", label="corrected-v3 test deciles")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed readmission rate")
        ax.set_title(
            "corrected-v3 reliability diagram (held-out test set)"
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(stage / OUTPUT_NAMES[6], dpi=150)
        plt.close(fig)

        still_existing = [path for path in destinations if path.exists()]
        _require(
            not still_existing,
            "Refusing to overwrite corrected-v3 evaluation output(s): "
            + ", ".join(str(path) for path in still_existing),
        )
        for name, destination in zip(OUTPUT_NAMES, destinations):
            os.rename(stage / name, destination)

    print(json.dumps(result, indent=2))
    print(f"Report: {destinations[0]}")


if __name__ == "__main__":
    main()
