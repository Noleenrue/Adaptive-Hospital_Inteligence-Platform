from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """
    Clinical features required by the readmission prediction model.
    """

    age_midpoint: float = Field(..., ge=0, le=120)

    gender: str
    race: str
    admission_type: str
    admission_source: str
    discharge_disposition: str

    length_of_stay: int = Field(..., ge=0)

    previous_inpatient_visits: int = Field(..., ge=0)
    previous_outpatient_visits: int = Field(..., ge=0)
    previous_emergency_visits: int = Field(..., ge=0)

    diagnosis_count: int = Field(..., ge=0)
    lab_procedure_count: int = Field(..., ge=0)
    procedure_count: int = Field(..., ge=0)
    medication_count: int = Field(..., ge=0)

    hba1c_result: str
    glucose_result: str
    insulin_status: str
    diabetes_medication: str

    primary_diagnosis_category: str
    secondary_diagnosis_category: str
    tertiary_diagnosis_category: str

    total_previous_visits: int = Field(..., ge=0)
    has_previous_inpatient_visit: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """
    Response returned by the readmission prediction API.
    """

    raw_probability: float = Field(..., ge=0, le=1)
    calibrated_probability: float = Field(..., ge=0, le=1)
    risk_tier: str
    prediction_id: int

    # Additive fields; existing consumers can ignore them.
    model_version: str | None = None
    artifact_id: str | None = None
