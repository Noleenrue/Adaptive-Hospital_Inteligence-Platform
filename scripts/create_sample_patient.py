"""
Create a sample patient and run the production readmission predictor.

This script is used as a simple end-to-end demonstration of the
Adaptive Hospital Intelligence Platform prediction pipeline.
"""

import pandas as pd

from src.features.feature_builder import select_model_features
from src.models.predictor import ReadmissionPredictor


def create_sample_patient() -> pd.DataFrame:
    """
    Create one example patient using the features expected by the model.
    """

    patient = {
        "age_midpoint": 65,
        "gender": "Female",
        "race": "Caucasian",

        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "discharge_disposition": "Discharged to home",

        "length_of_stay": 5,

        "previous_inpatient_visits": 1,
        "previous_outpatient_visits": 2,
        "previous_emergency_visits": 1,

        "diagnosis_count": 7,
        "lab_procedure_count": 45,
        "procedure_count": 1,
        "medication_count": 15,

        "hba1c_result": "None",
        "glucose_result": "None",

        "insulin_status": "Steady",
        "diabetes_medication": "Yes",

        "primary_diagnosis_category": "Circulatory",
        "secondary_diagnosis_category": "Diabetes",
        "tertiary_diagnosis_category": "Other",

        # Engineered features
        "total_previous_visits": 4,
        "has_previous_inpatient_visit": 1,
    }

    return pd.DataFrame([patient])


def main():
    print("=" * 60)
    print("ADAPTIVE HOSPITAL INTELLIGENCE PLATFORM")
    print("Sample Patient Prediction")
    print("=" * 60)

    # Create patient
    patient_df = create_sample_patient()

    # Ensure exact model feature structure
    patient_df = select_model_features(patient_df)

    print("\nPatient data:")
    print(patient_df.T)

    # Load production predictor
    predictor = ReadmissionPredictor()

    # Generate prediction
    result = predictor.predict(patient_df)

    print("\nPrediction result:")
    print(result)


if __name__ == "__main__":
    main()
