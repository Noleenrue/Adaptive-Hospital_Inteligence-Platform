from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicalEncounter:
    """
    Canonical representation of a hospital encounter
    used throughout the AHIP platform.
    """

    # Identity
    patient_id: str
    encounter_id: str

    # Demographics
    age: Optional[str] = None
    gender: Optional[str] = None
    race: Optional[str] = None

    # Encounter information
    admission_type: Optional[str] = None
    admission_source: Optional[str] = None
    discharge_disposition: Optional[str] = None
    length_of_stay: Optional[int] = None

    # Previous healthcare utilization
    previous_inpatient_visits: Optional[int] = None
    previous_outpatient_visits: Optional[int] = None
    previous_emergency_visits: Optional[int] = None

    # Clinical information
    primary_diagnosis: Optional[str] = None
    secondary_diagnosis: Optional[str] = None
    tertiary_diagnosis: Optional[str] = None
    diagnosis_count: Optional[int] = None

    # Treatment / hospital activity
    lab_procedure_count: Optional[int] = None
    procedure_count: Optional[int] = None
    medication_count: Optional[int] = None

    # Diabetes-related information
    hba1c_result: Optional[str] = None
    glucose_result: Optional[str] = None
    insulin_status: Optional[str] = None
    diabetes_medication: Optional[str] = None

    # Outcome
    readmitted_30_days: Optional[bool] = None
