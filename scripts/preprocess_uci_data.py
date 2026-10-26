"""Build a candidate processed dataset from the UCI Diabetes CSV files.

Run from the project root with:
    python scripts/preprocess_uci_data.py [--run-id RUN_ID]

Outputs are written only to data/processed/candidates/<run-id>/.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.uci_diabetes import UCIDiabetesAdapter  # noqa: E402
from src.features.clinical_features import ClinicalFeatureEngineer  # noqa: E402
from src.features.feature_builder import MODEL_FEATURES  # noqa: E402
from src.ingestion.reference_data import load_uci_id_mappings  # noqa: E402
from src.preprocessing.clinical import ClinicalPreprocessor  # noqa: E402
from src.preprocessing.lab_encoding import (  # noqa: E402
    INPUT_MAPPING_ID,
    LAB_COLUMNS,
    NOT_MEASURED,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CANDIDATE_DIR = PROCESSED_DIR / "candidates"
RAW_DATA_PATH = RAW_DIR / "diabetic_data.csv"
MAPPING_PATH = RAW_DIR / "IDS_mapping.csv"
REFERENCE_PATH = PROCESSED_DIR / "clinical_features.csv"
OUTPUT_FILENAME = "clinical_features.csv"
REPORT_JSON_FILENAME = "validation_report.json"
REPORT_MARKDOWN_FILENAME = "validation_report.md"
TARGET = "readmitted_30_days"
GROUP = "patient_id"
ENCOUNTER = "encounter_id"

OUTPUT_COLUMNS = [
    "patient_id",
    "encounter_id",
    "age_midpoint",
    "gender",
    "race",
    "admission_type",
    "admission_source",
    "discharge_disposition",
    "length_of_stay",
    "previous_inpatient_visits",
    "previous_outpatient_visits",
    "previous_emergency_visits",
    "primary_diagnosis",
    "secondary_diagnosis",
    "tertiary_diagnosis",
    "diagnosis_count",
    "lab_procedure_count",
    "procedure_count",
    "medication_count",
    "hba1c_result",
    "glucose_result",
    "insulin_status",
    "diabetes_medication",
    TARGET,
    "primary_diagnosis_category",
    "secondary_diagnosis_category",
    "tertiary_diagnosis_category",
    "total_previous_visits",
    "has_previous_inpatient_visit",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _id_digest(values) -> str:
    normalized = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def build_processed_features(raw_data: pd.DataFrame, mappings) -> pd.DataFrame:
    """Apply the existing source adapter, canonical preprocessing and features."""

    encounters = UCIDiabetesAdapter(mappings).transform(raw_data)
    canonical = ClinicalPreprocessor().transform(encounters)
    engineered = ClinicalFeatureEngineer().transform(canonical)

    missing = [column for column in OUTPUT_COLUMNS if column not in engineered]
    if missing:
        raise ValueError(f"Preprocessing did not create required columns: {missing}")

    output = engineered.loc[:, OUTPUT_COLUMNS].copy()
    output[TARGET] = output[TARGET].astype(bool)
    return output


def _target_distribution(frame: pd.DataFrame) -> dict:
    if TARGET not in frame:
        return {}
    counts = frame[TARGET].astype(str).value_counts(dropna=False)
    return {str(label): int(count) for label, count in sorted(counts.items())}


def _duplicate_count(frame: pd.DataFrame, column: str) -> dict:
    if column not in frame:
        return {"column_present": False, "duplicate_rows": None}
    duplicates = frame[column].duplicated(keep="first")
    return {
        "column_present": True,
        "duplicate_rows": int(duplicates.sum()),
        "unique_values": int(frame[column].nunique(dropna=False)),
    }


def _compare_frames(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    candidate_columns = list(candidate.columns)
    reference_columns = list(reference.columns)
    common_columns = [
        column
        for column in candidate_columns
        if column in reference_columns and column != ENCOUNTER
    ]

    candidate_ids_unique = (
        ENCOUNTER in candidate
        and not candidate[ENCOUNTER].duplicated().any()
    )
    reference_ids_unique = (
        ENCOUNTER in reference
        and not reference[ENCOUNTER].duplicated().any()
    )
    can_compare_rows = (
        candidate_ids_unique
        and reference_ids_unique
        and set(candidate[ENCOUNTER]) == set(reference[ENCOUNTER])
    )

    row_comparison = {
        "performed": bool(can_compare_rows),
        "rows_with_any_value_mismatch": None,
        "mismatched_values_by_column": {},
    }

    if can_compare_rows:
        left = candidate.set_index(ENCOUNTER).sort_index()
        right = reference.set_index(ENCOUNTER).sort_index()
        mismatches = {}
        row_mismatch = pd.Series(False, index=left.index)

        for column in common_columns:
            left_values = left[column]
            right_values = right[column]
            equal = left_values.eq(right_values) | (
                left_values.isna() & right_values.isna()
            )
            mismatch_count = int((~equal).sum())
            mismatches[column] = mismatch_count
            row_mismatch |= ~equal

        row_comparison.update({
            "rows_with_any_value_mismatch": int(row_mismatch.sum()),
            "mismatched_values_by_column": mismatches,
        })

    candidate_ids = (
        set(candidate[GROUP]) if GROUP in candidate else set()
    )
    reference_ids = (
        set(reference[GROUP]) if GROUP in reference else set()
    )
    candidate_encounters = (
        set(candidate[ENCOUNTER]) if ENCOUNTER in candidate else set()
    )
    reference_encounters = (
        set(reference[ENCOUNTER]) if ENCOUNTER in reference else set()
    )

    return {
        "reference_available": True,
        "row_counts_match": len(candidate) == len(reference),
        "candidate_rows": len(candidate),
        "reference_rows": len(reference),
        "columns_match_exactly": candidate_columns == reference_columns,
        "candidate_columns": candidate_columns,
        "reference_columns": reference_columns,
        "dtypes_match_exactly": (
            candidate.dtypes.astype(str).to_dict()
            == reference.dtypes.astype(str).to_dict()
        ),
        "candidate_dtypes": {
            column: str(dtype)
            for column, dtype in candidate.dtypes.items()
        },
        "reference_dtypes": {
            column: str(dtype)
            for column, dtype in reference.dtypes.items()
        },
        "target_distribution_matches": (
            _target_distribution(candidate)
            == _target_distribution(reference)
        ),
        "candidate_target_distribution": _target_distribution(candidate),
        "reference_target_distribution": _target_distribution(reference),
        "patient_ids": {
            "set_matches": candidate_ids == reference_ids,
            "candidate_unique_count": len(candidate_ids),
            "reference_unique_count": len(reference_ids),
            "candidate_sorted_sha256": _id_digest(candidate_ids),
            "reference_sorted_sha256": _id_digest(reference_ids),
        },
        "encounter_ids": {
            "set_matches": candidate_encounters == reference_encounters,
            "candidate_unique_count": len(candidate_encounters),
            "reference_unique_count": len(reference_encounters),
            "candidate_sorted_sha256": _id_digest(candidate_encounters),
            "reference_sorted_sha256": _id_digest(reference_encounters),
        },
        "duplicates": {
            "patient_id": {
                "candidate": _duplicate_count(candidate, GROUP),
                "reference": _duplicate_count(reference, GROUP),
            },
            "encounter_id": {
                "candidate": _duplicate_count(candidate, ENCOUNTER),
                "reference": _duplicate_count(reference, ENCOUNTER),
            },
        },
        "row_values": row_comparison,
    }


def compare_candidate_to_reference(
    candidate: pd.DataFrame, reference_path: Path
) -> dict:
    reference_path = Path(reference_path)
    if not reference_path.is_file():
        return {
            "reference_available": False,
            "reference_path": str(reference_path),
            "reason": "Reference processed dataset was not found.",
        }
    reference = pd.read_csv(reference_path)
    result = _compare_frames(candidate, reference)
    result["reference_path"] = _display_path(reference_path)
    result["reference_sha256"] = sha256_file(reference_path)
    return result


def _comparison_matches(comparison: dict) -> bool | None:
    if not comparison.get("reference_available"):
        return None
    checks = [
        comparison["row_counts_match"],
        comparison["columns_match_exactly"],
        comparison["dtypes_match_exactly"],
        comparison["target_distribution_matches"],
        comparison["patient_ids"]["set_matches"],
        comparison["encounter_ids"]["set_matches"],
        comparison["row_values"]["performed"],
        comparison["row_values"]["rows_with_any_value_mismatch"] == 0,
        all(
            value["candidate"]["duplicate_rows"]
            == value["reference"]["duplicate_rows"]
            for value in comparison["duplicates"].values()
        ),
    ]
    return all(checks)


def _validate_run_id(run_id: str) -> str:
    if (
        run_id in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id)
    ):
        raise ValueError(
            "run-id must be 1-80 letters, digits, dots, underscores, or "
            "hyphens, and must not be '.' or '..'."
        )
    return run_id


def _markdown_report(report: dict) -> str:
    comparison = report["comparison"]
    lines = [
        "# UCI preprocessing candidate validation",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Candidate rows: {report['output']['rows']}",
        f"- Expired-discharge rows retained in processed candidate: "
        f"{report['preprocessing']['expired_discharge_rows_retained']}",
        f"- Corrected-v3 exporter exclusion rule: "
        f"{report['preprocessing']['training_exclusion_rule']}",
        f"- Reference comparison: "
        f"{'exact match' if report['comparison_matches'] is True else 'differences found' if report['comparison_matches'] is False else 'not available'}",
        "",
        "## Input and output hashes",
        "",
        f"- Raw data SHA-256: `{report['inputs']['raw_data_sha256']}`",
        f"- Mapping SHA-256: `{report['inputs']['mapping_sha256']}`",
        f"- Candidate CSV SHA-256: `{report['output']['sha256']}`",
        "",
        "## Comparison summary",
        "",
    ]

    if not comparison.get("reference_available"):
        lines.append(comparison.get("reason", "Reference unavailable."))
    else:
        lines.extend([
            f"- Row counts match: {comparison['row_counts_match']}",
            f"- Column names and order match: "
            f"{comparison['columns_match_exactly']}",
            f"- CSV-inferred dtypes match: "
            f"{comparison['dtypes_match_exactly']}",
            f"- Target distribution matches: "
            f"{comparison['target_distribution_matches']}",
            f"- Patient identifier sets match: "
            f"{comparison['patient_ids']['set_matches']}",
            f"- Encounter identifier sets match: "
            f"{comparison['encounter_ids']['set_matches']}",
            f"- Row-value comparison performed: "
            f"{comparison['row_values']['performed']}",
            f"- Rows with any value mismatch: "
            f"{comparison['row_values']['rows_with_any_value_mismatch']}",
            "",
            "Mismatched values by column:",
            "",
        ])
        mismatches = comparison["row_values"].get(
            "mismatched_values_by_column", {}
        )
        if mismatches:
            lines.extend(
                f"- `{column}`: {count}"
                for column, count in mismatches.items()
                if count
            )
            if not any(mismatches.values()):
                lines.append("- None")
        else:
            lines.append("- Not calculated (encounter identifiers were not unique/equal).")

    lines.extend([
        "",
        "## Reproducibility statement",
        "",
        (
            "The candidate matches the existing processed CSV under the "
            "recorded checks. This establishes output agreement for these "
            "local inputs and this implementation; it does not establish "
            "that these raw files are byte-identical to the historical raw "
            "inputs unless their historical hashes are independently known."
            if report["comparison_matches"] is True
            else
            "Exact preprocessing reproducibility has not been demonstrated. "
            "See the comparison details above; transformations were not "
            "changed to force a match."
        ),
        "",
    ])
    return "\n".join(lines)


def run_preprocessing(
    raw_data_path: Path,
    mapping_path: Path,
    reference_path: Path,
    run_id: str,
    candidate_root: Path = CANDIDATE_DIR,
) -> Path:
    run_id = _validate_run_id(run_id)
    candidate_root = Path(candidate_root)
    candidate_dir = candidate_root / run_id
    if candidate_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate output: {candidate_dir}"
        )

    raw_data_path = Path(raw_data_path)
    mapping_path = Path(mapping_path)
    raw_data = pd.read_csv(raw_data_path)
    mappings = load_uci_id_mappings(mapping_path)
    processed = build_processed_features(raw_data, mappings)
    expired = processed["discharge_disposition"].astype(str).str.contains(
        "Expired", case=False, na=False
    )
    candidate_dir.mkdir(parents=True, exist_ok=False)
    output_path = candidate_dir / OUTPUT_FILENAME
    processed.to_csv(output_path, index=False)
    persisted = pd.read_csv(output_path)
    comparison = compare_candidate_to_reference(persisted, reference_path)
    comparison_matches = _comparison_matches(comparison)

    report = {
        "run_id": run_id,
        "status": "completed",
        "comparison_matches": comparison_matches,
        "inputs": {
            "raw_data_path": _display_path(raw_data_path),
            "raw_data_sha256": sha256_file(raw_data_path),
            "mapping_path": _display_path(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
        },
        "preprocessing": {
            "adapter": "UCIDiabetesAdapter",
            "reference_data_loader": "load_uci_id_mappings",
            "clinical_preprocessor": "ClinicalPreprocessor",
            "feature_engineer": "ClinicalFeatureEngineer",
            "model_feature_schema": list(MODEL_FEATURES),
            "output_columns": OUTPUT_COLUMNS,
            "target": {
                "column": TARGET,
                "definition": "readmitted == '<30'",
                "positive_value": True,
            },
            "group_column": GROUP,
            "identifier_columns_preserved": [GROUP, ENCOUNTER],
            "lab_input_mapping": {
                "id": INPUT_MAPPING_ID,
                "label": NOT_MEASURED,
                "applied_to_processed_values": False,
                "semantics": (
                    "Missing/unmeasured HbA1c and glucose remain missing "
                    "in the processed CSV. The corrected-v3 training and "
                    "serving mapping is applied downstream by the model "
                    "exporter/predictor."
                ),
                "columns": list(LAB_COLUMNS),
            },
            "expired_discharge_rows_retained": int(expired.sum()),
            "training_exclusion_rule": (
                "The corrected-v3 exporter excludes rows whose "
                "discharge_disposition contains 'Expired' "
                "(case-insensitive) before splitting."
            ),
            "randomness": "None; preprocessing preserves source row order.",
        },
        "output": {
            "path": _display_path(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(processed),
            "columns": list(processed.columns),
            "dtypes": {
                column: str(dtype)
                for column, dtype in persisted.dtypes.items()
            },
            "target_distribution": _target_distribution(persisted),
            "patient_ids": {
                "unique_count": int(persisted[GROUP].nunique()),
                "sorted_sha256": _id_digest(persisted[GROUP].unique()),
                "duplicates": _duplicate_count(persisted, GROUP),
            },
            "encounter_ids": {
                "unique_count": int(persisted[ENCOUNTER].nunique()),
                "sorted_sha256": _id_digest(persisted[ENCOUNTER].unique()),
                "duplicates": _duplicate_count(persisted, ENCOUNTER),
            },
        },
        "comparison": comparison,
    }
    (candidate_dir / REPORT_JSON_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (candidate_dir / REPORT_MARKDOWN_FILENAME).write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return candidate_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("uci-%Y%m%dT%H%M%SZ"),
        help="Unique candidate directory name; existing output is never overwritten.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the already-downloaded UCI diabetic_data.csv.",
    )
    parser.add_argument(
        "--mapping-data",
        type=Path,
        default=MAPPING_PATH,
        help="Path to the already-downloaded UCI IDS_mapping.csv.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_PATH,
        help="Optional comparison source; a missing path is reported, not generated.",
    )
    args = parser.parse_args()

    candidate_dir = run_preprocessing(
        args.raw_data,
        args.mapping_data,
        args.reference,
        args.run_id,
    )
    print(f"Candidate output: {candidate_dir}")
    print(f"Validation report: {candidate_dir / REPORT_MARKDOWN_FILENAME}")


if __name__ == "__main__":
    main()
