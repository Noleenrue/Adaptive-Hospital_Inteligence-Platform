import pandas as pd

from src.features.diagnosis import categorize_diagnosis


class ClinicalFeatureEngineer:
    """
    Creates derived clinical features for machine-learning models.

    This class expects data that has already passed through
    ClinicalPreprocessor.
    """

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy so we do not modify the original DataFrame.
        df = data.copy()

        # ---------------------------------------------------------
        # 1. Diagnosis categories
        # ---------------------------------------------------------
        diagnosis_columns = [
            "primary_diagnosis",
            "secondary_diagnosis",
            "tertiary_diagnosis",
        ]

        for column in diagnosis_columns:
            if column in df.columns:
                df[f"{column}_category"] = (
                    df[column].apply(categorize_diagnosis)
                )

        # ---------------------------------------------------------
        # 2. Previous healthcare utilization
        # ---------------------------------------------------------
        utilization_columns = [
            "previous_inpatient_visits",
            "previous_outpatient_visits",
            "previous_emergency_visits",
        ]

        available_utilization_columns = [
            column
            for column in utilization_columns
            if column in df.columns
        ]

        if available_utilization_columns:
            df["total_previous_visits"] = (
                df[available_utilization_columns]
                .fillna(0)
                .sum(axis=1)
            )

        # ---------------------------------------------------------
        # 3. Whether patient had a previous inpatient admission
        # ---------------------------------------------------------
        if "previous_inpatient_visits" in df.columns:
            df["has_previous_inpatient_visit"] = (
                df["previous_inpatient_visits"]
                .fillna(0)
                .gt(0)
                .astype(int)
            )

        return df
