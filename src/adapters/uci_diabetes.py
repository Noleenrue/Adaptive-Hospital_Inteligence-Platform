from typing import List

import pandas as pd

from src.adapters.base import BaseHospitalAdapter
from src.schemas.clinical import ClinicalEncounter


class UCIDiabetesAdapter(BaseHospitalAdapter):
    """
    Adapter for the UCI Diabetes 130-US Hospitals dataset.

    Converts UCI-specific column names and values into
    AHIP's canonical ClinicalEncounter schema.
    """

    REQUIRED_COLUMNS = {
        "patient_nbr",
        "encounter_id",
        "age",
        "gender",
        "time_in_hospital",
        "number_inpatient",
        "number_outpatient",
        "number_emergency",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "A1Cresult",
        "max_glu_serum",
        "insulin",
        "diabetesMed",
        "readmitted",
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id",
    }

    def __init__(self, mappings=None):
        """
        Initialize the adapter with optional reference-data mappings.
        """
        self.mappings = mappings or {}

    def validate_source(self, data: pd.DataFrame) -> bool:
        return self.REQUIRED_COLUMNS.issubset(data.columns)

    @staticmethod
    def _clean_value(value):
        """
        Convert UCI missing-value markers into Python None.
        """

        if pd.isna(value) or value == "?":
            return None

        return value

    def _lookup(self, mapping_name: str, code):
        if pd.isna(code):
            return None

        mapping = self.mappings.get(mapping_name, {})

        try:
            normalized_code = int(code)
        except (TypeError, ValueError):
            return f"Unmapped:{code}"

        value = mapping.get(normalized_code)

        if value is None or pd.isna(value):
            return f"Unmapped:{normalized_code}"

        value = str(value).strip()

        if not value or value.lower() == "nan":
            return f"Unmapped:{normalized_code}"

        return value

    def transform(self, data: pd.DataFrame) -> List[ClinicalEncounter]:
        """
        Transform UCI Diabetes records into AHIP ClinicalEncounter objects.
        """

        if not self.validate_source(data):
            missing_columns = self.REQUIRED_COLUMNS - set(data.columns)

            raise ValueError(
                f"UCI dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        encounters = []

        for _, row in data.iterrows():
            encounter = ClinicalEncounter(
                patient_id=str(row["patient_nbr"]),
                encounter_id=str(row["encounter_id"]),

                age=self._clean_value(row["age"]),
                gender=self._clean_value(row["gender"]),
                race=self._clean_value(row.get("race")),

                admission_type=self._lookup(
                    "admission_type",
                    row["admission_type_id"],
                ),

                admission_source=self._lookup(
                    "admission_source",
                    row["admission_source_id"],
                ),

                discharge_disposition=self._lookup(
                    "discharge_disposition",
                    row["discharge_disposition_id"],
                ),

                length_of_stay=int(row["time_in_hospital"]),

                previous_inpatient_visits=int(row["number_inpatient"]),
                previous_outpatient_visits=int(row["number_outpatient"]),
                previous_emergency_visits=int(row["number_emergency"]),

                primary_diagnosis=self._clean_value(row["diag_1"]),
                secondary_diagnosis=self._clean_value(row["diag_2"]),
                tertiary_diagnosis=self._clean_value(row["diag_3"]),
                diagnosis_count=int(row["number_diagnoses"]),

                lab_procedure_count=int(row["num_lab_procedures"]),
                procedure_count=int(row["num_procedures"]),
                medication_count=int(row["num_medications"]),

                hba1c_result=self._clean_value(row["A1Cresult"]),
                glucose_result=self._clean_value(row["max_glu_serum"]),
                insulin_status=self._clean_value(row["insulin"]),
                diabetes_medication=self._clean_value(row["diabetesMed"]),

                readmitted_30_days=(row["readmitted"] == "<30"),
            )

            encounters.append(encounter)

        return encounters
