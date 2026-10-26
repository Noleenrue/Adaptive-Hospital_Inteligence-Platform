import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.preprocess_uci_data import (
    OUTPUT_COLUMNS,
    TARGET,
    build_processed_features,
    run_preprocessing,
)
from src.features.feature_builder import MODEL_FEATURES
from src.ingestion.reference_data import load_uci_id_mappings


def _mapping_frame():
    return pd.DataFrame(
        [
            ("1", "Emergency"),
            ("2", "Urgent"),
            ("discharge_disposition_id", None),
            ("1", "Discharged to home"),
            ("2", "Expired"),
            ("admission_source_id", None),
            ("1", "Emergency Room"),
            ("2", "Physician Referral"),
        ],
        columns=["admission_type_id", "description"],
    )


def _raw_frame():
    return pd.DataFrame(
        {
            "patient_nbr": [100, 100, 200, 300],
            "encounter_id": [1001, 1002, 2001, 3001],
            "age": ["[60-70)", "[70-80)", "[50-60)", "[40-50)"],
            "gender": ["Female", "Male", "Female", "Unknown/Invalid"],
            "race": ["Caucasian", "AfricanAmerican", "?", "Hispanic"],
            "admission_type_id": [1, 2, 1, 1],
            "admission_source_id": [1, 2, 1, 99],
            "discharge_disposition_id": [1, 1, 2, 1],
            "time_in_hospital": [3, 5, 2, 1],
            "number_inpatient": [0, 1, 0, 2],
            "number_outpatient": [1, 0, 0, 1],
            "number_emergency": [0, 1, 0, 0],
            "diag_1": ["250.13", "414.00", "?", "E849.7"],
            "diag_2": ["401.9", "250.00", "428.0", None],
            "diag_3": ["V45.81", "?", "250.00", "486"],
            "number_diagnoses": [3, 4, 2, 3],
            "num_lab_procedures": [30, 40, 20, 10],
            "num_procedures": [1, 0, 2, 0],
            "num_medications": [8, 10, 5, 4],
            "A1Cresult": [None, "Norm", ">8", "?"],
            "max_glu_serum": [None, ">200", "Norm", None],
            "insulin": ["Steady", "No", "Up", "Down"],
            "diabetesMed": ["Yes", "Yes", "No", "Yes"],
            "readmitted": ["<30", ">30", "NO", "<30"],
        }
    )


def _write_inputs(tmp_path):
    raw_path = tmp_path / "raw.csv"
    mapping_path = tmp_path / "mapping.csv"
    _raw_frame().to_csv(raw_path, index=False)
    _mapping_frame().to_csv(mapping_path, index=False)
    return raw_path, mapping_path


def test_candidate_dataset_preserves_target_ids_exclusions_and_features(tmp_path):
    _, mapping_path = _write_inputs(tmp_path)
    mappings = load_uci_id_mappings(mapping_path)

    result = build_processed_features(_raw_frame(), mappings)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert list(result[MODEL_FEATURES].columns) == MODEL_FEATURES
    assert result["patient_id"].tolist() == ["100", "100", "200", "300"]
    assert result["encounter_id"].tolist() == ["1001", "1002", "2001", "3001"]
    assert result[TARGET].tolist() == [True, False, False, True]
    assert result.loc[2, "discharge_disposition"] == "Expired"
    assert result["admission_type"].tolist() == [
        "Emergency", "Urgent", "Emergency", "Emergency"
    ]
    assert result["admission_source"].tolist() == [
        "Emergency Room", "Physician Referral", "Emergency Room",
        "Unmapped:99",
    ]


def test_candidate_preserves_unmeasured_labs_and_creates_derived_features(
    tmp_path,
):
    _, mapping_path = _write_inputs(tmp_path)
    mappings = load_uci_id_mappings(mapping_path)

    result = build_processed_features(_raw_frame(), mappings)

    assert pd.isna(result.loc[0, "hba1c_result"])
    assert pd.isna(result.loc[0, "glucose_result"])
    assert result.loc[0, "primary_diagnosis_category"] == "Diabetes"
    assert result.loc[1, "primary_diagnosis_category"] == "Circulatory"
    assert result["total_previous_visits"].tolist() == [1, 2, 0, 3]
    assert result["has_previous_inpatient_visit"].tolist() == [0, 1, 0, 1]


def test_candidate_write_compares_existing_reference_and_records_hashes(
    tmp_path,
):
    raw_path, mapping_path = _write_inputs(tmp_path)
    mappings = load_uci_id_mappings(mapping_path)
    reference = build_processed_features(_raw_frame(), mappings)
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)
    candidate_root = tmp_path / "candidates"

    output_dir = run_preprocessing(
        raw_path,
        mapping_path,
        reference_path,
        "synthetic-run",
        candidate_root=candidate_root,
    )

    candidate = pd.read_csv(output_dir / "clinical_features.csv")
    report = json.loads(
        (output_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    assert candidate_root in output_dir.parents
    assert report["comparison_matches"] is True
    assert report["comparison"]["row_values"]["rows_with_any_value_mismatch"] == 0
    assert report["inputs"]["raw_data_sha256"]
    assert report["inputs"]["mapping_sha256"]
    assert report["output"]["sha256"]
    assert report["output"]["rows"] == 4
    assert report["preprocessing"]["expired_discharge_rows_retained"] == 1


def test_existing_candidate_is_never_overwritten(tmp_path):
    raw_path, mapping_path = _write_inputs(tmp_path)
    mappings = load_uci_id_mappings(mapping_path)
    reference_path = tmp_path / "reference.csv"
    build_processed_features(_raw_frame(), mappings).to_csv(
        reference_path, index=False
    )
    candidate_root = tmp_path / "candidates"

    output_dir = run_preprocessing(
        raw_path,
        mapping_path,
        reference_path,
        "no-overwrite",
        candidate_root=candidate_root,
    )
    original_output = (output_dir / "clinical_features.csv").read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_preprocessing(
            raw_path,
            mapping_path,
            reference_path,
            "no-overwrite",
            candidate_root=candidate_root,
        )

    assert (output_dir / "clinical_features.csv").read_bytes() == original_output


@pytest.mark.parametrize("run_id", ["..", "../escape", "bad/name", ""])
def test_run_id_cannot_escape_candidate_directory(tmp_path, run_id):
    raw_path, mapping_path = _write_inputs(tmp_path)
    candidate_root = tmp_path / "candidates"

    with pytest.raises(ValueError, match="run-id"):
        run_preprocessing(
            raw_path,
            mapping_path,
            tmp_path / "missing-reference.csv",
            run_id,
            candidate_root=candidate_root,
        )

    assert not candidate_root.exists()
