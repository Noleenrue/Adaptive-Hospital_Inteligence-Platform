import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate_corrected_v3 import (
    EXPECTED_ARTIFACT_ID,
    EXPECTED_VERSION,
    _verify_bundle_and_data,
    calibration_slope_intercept,
    cluster_resample_indices,
    patient_cluster_bootstrap,
    reconstruct_partitions,
    threshold_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_corrected_v3_manifest_data_and_original_split_are_verified():
    predictor, manifest, data_path, data_hash = _verify_bundle_and_data()
    df = pd.read_csv(data_path)
    eligible, partitions, _ = reconstruct_partitions(df, manifest)

    assert predictor.model_version == EXPECTED_VERSION
    assert predictor.artifact_id == EXPECTED_ARTIFACT_ID
    assert data_hash == manifest["data"]["source_sha256"]
    assert len(eligible) == manifest["data"]["eligible_encounters"]
    assert {
        name: (len(X), int(groups.nunique()))
        for name, (X, _, groups) in partitions.items()
    } == {
        name: (
            manifest["split"][f"{name}_encounters"],
            manifest["split"][f"{name}_patients"],
        )
        for name in ("train", "calibration", "test")
    }


def test_partition_reconstruction_rejects_manifest_count_mismatch():
    _, manifest, data_path, _ = _verify_bundle_and_data()
    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["split"]["test_patients"] += 1

    with pytest.raises(RuntimeError, match="test patient count"):
        reconstruct_partitions(pd.read_csv(data_path), bad_manifest)


def test_partition_reconstruction_rejects_manifest_seed_mismatch():
    _, manifest, data_path, _ = _verify_bundle_and_data()
    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["split"]["random_state"] += 1

    with pytest.raises(RuntimeError, match="random seed"):
        reconstruct_partitions(pd.read_csv(data_path), bad_manifest)


def test_threshold_metrics_match_confusion_counts_and_scores():
    result = threshold_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.15, 0.9]),
        0.15,
    )

    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (
        1, 1, 0, 2
    )
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1)
    assert result["f1"] == pytest.approx(0.8)


def test_cluster_resampling_keeps_each_patient_encounters_together():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    sampled_groups, indices = cluster_resample_indices(
        groups, np.random.default_rng(12)
    )
    expected_counts = {
        patient: int(np.sum(sampled_groups == patient))
        for patient in np.unique(groups)
    }

    for patient, multiplicity in expected_counts.items():
        assert np.sum(groups[indices] == patient) == (
            np.sum(groups == patient) * multiplicity
        )
    assert len(sampled_groups) == len(np.unique(groups))


def test_bootstrap_reports_skipped_single_class_samples_and_valid_count():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8])
    groups = np.array(["negative", "negative", "positive", "positive"])

    result = patient_cluster_bootstrap(
        y_true, probabilities, groups, 0.5, n_resamples=40, seed=31
    )

    assert result["valid_resamples"] > 0
    assert result["single_class_resamples_skipped"] > 0
    assert (
        result["valid_resamples"]
        + result["single_class_resamples_skipped"]
        == result["requested_resamples"]
    )
    for interval in result["intervals"].values():
        assert interval["valid_resamples"] == result["valid_resamples"]
        assert interval["lower_95"] <= interval["upper_95"]


def test_calibration_regression_is_diagnostic_and_returns_slope_intercept():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probability = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.6])

    result = calibration_slope_intercept(y_true, probability)

    assert np.isfinite(result["slope"])
    assert np.isfinite(result["intercept"])
