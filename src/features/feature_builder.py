import pandas as pd


MODEL_FEATURES = [
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
    "diagnosis_count",
    "lab_procedure_count",
    "procedure_count",
    "medication_count",
    "hba1c_result",
    "glucose_result",
    "insulin_status",
    "diabetes_medication",
    "primary_diagnosis_category",
    "secondary_diagnosis_category",
    "tertiary_diagnosis_category",
    "total_previous_visits",
    "has_previous_inpatient_visit",
]


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the exact 23 features expected by the readmission model.

    Raises
    ------
    ValueError
        If one or more required model features are missing.
    """

    missing_features = [
        feature for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required model features: {missing_features}"
        )

    return df[MODEL_FEATURES].copy()
