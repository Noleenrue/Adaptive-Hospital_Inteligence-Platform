from typing import Optional, List

import pandas as pd

from src.schemas.clinical import ClinicalEncounter

AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}


def age_group_to_midpoint(age_group: Optional[str]) -> Optional[int]:
    """
    Convert an age interval into its midpoint.

    Example:
        '[60-70)' -> 65

    The returned value is a derived approximation and does not
    represent the patient's exact age.
    """

    if age_group is None:
        return None

    return AGE_MIDPOINTS.get(age_group)

class ClinicalPreprocessor:
    """
    Converts canonical ClinicalEncounter objects into
    an analytical pandas DataFrame.
    """

    def transform(
        self,
        encounters: List[ClinicalEncounter]
    ) -> pd.DataFrame:

        records = []

        for encounter in encounters:

            record = {
                # Identifiers
                "patient_id": encounter.patient_id,
                "encounter_id": encounter.encounter_id,

                # Demographics
                "age_midpoint": age_group_to_midpoint(encounter.age),
                "gender": encounter.gender,
                "race": encounter.race,

                # Hospital encounter
                "admission_type": encounter.admission_type,
                "admission_source": encounter.admission_source,
                "discharge_disposition": encounter.discharge_disposition,
                "length_of_stay": encounter.length_of_stay,

                # Previous healthcare utilization
                "previous_inpatient_visits": encounter.previous_inpatient_visits,
                "previous_outpatient_visits": encounter.previous_outpatient_visits,
                "previous_emergency_visits": encounter.previous_emergency_visits,

                # Diagnosis
                "primary_diagnosis": encounter.primary_diagnosis,
                "secondary_diagnosis": encounter.secondary_diagnosis,
                "tertiary_diagnosis": encounter.tertiary_diagnosis,
                "diagnosis_count": encounter.diagnosis_count,

                # Hospital activity
                "lab_procedure_count": encounter.lab_procedure_count,
                "procedure_count": encounter.procedure_count,
                "medication_count": encounter.medication_count,

                # Diabetes information
                "hba1c_result": encounter.hba1c_result,
                "glucose_result": encounter.glucose_result,
                "insulin_status": encounter.insulin_status,
                "diabetes_medication": encounter.diabetes_medication,

                # Prediction target
                "readmitted_30_days": encounter.readmitted_30_days,
            }

            records.append(record)

        return pd.DataFrame(records)
