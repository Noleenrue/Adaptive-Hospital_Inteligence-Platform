
from src.explainability.shap_explainer import ReadmissionSHAPExplainer
from src.models.predictor import ReadmissionPredictor

def test_shap_explanation():
    predictor = ReadmissionPredictor()

    patient = {
        "age_midpoint": 65.0,
        "gender": "Female",
        "race": "Caucasian",
        "admission_type": "Emergency",
        "admission_source": "Emergency Room",
        "discharge_disposition": "Discharged to home",
        "length_of_stay": 4,
        "previous_inpatient_visits": 1,
        "previous_outpatient_visits": 0,
        "previous_emergency_visits": 0,
        "diagnosis_count": 5,
        "lab_procedure_count": 40,
        "procedure_count": 1,
        "medication_count": 12,
        "hba1c_result": "None",
        "glucose_result": "None",
        "insulin_status": "No",
        "diabetes_medication": "Yes",
        "primary_diagnosis_category": "Circulatory",
        "secondary_diagnosis_category": "Diabetes",
        "tertiary_diagnosis_category": "Other",
        "total_previous_visits": 1,
        "has_previous_inpatient_visit": 1
    }

    explainer = ReadmissionSHAPExplainer(predictor)

    result = explainer.explain(patient, top_n=10)

    assert len(result["top_features"]) > 0
    assert len(result["top_features"]) <= 10

    for feature in result["top_features"]:
        assert "feature" in feature
        assert "shap_value" in feature

    print("\nSHAP explanation generated successfully!")

    for feature in result["top_features"]:
        print(
            feature["feature"],
            round(feature["shap_value"], 4)
        )


if __name__ == "__main__":
    test_shap_explanation()
