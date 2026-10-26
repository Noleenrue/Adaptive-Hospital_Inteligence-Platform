import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from database.repository import save_prediction
from database.repository import get_prediction_history


from api.schemas import PatientFeatures, PredictionResponse
from src.models.predictor import ReadmissionPredictor
from src.explainability.shap_explainer import ReadmissionSHAPExplainer


app = FastAPI(
    title="Adaptive Hospital Intelligence Platform",
    description=(
        "API for estimating 30-day hospital readmission risk."
    ),
    version="1.0.0",
)


# Load model artifacts once when the API starts.
predictor = ReadmissionPredictor()

_explainer = None


def _get_explainer():
    """Explainer bound to the same loaded predictor used for /predict."""

    global _explainer

    if _explainer is None:
        _explainer = ReadmissionSHAPExplainer(predictor)

    return _explainer

@app.get("/")
def root():
    return {
        "service": "Adaptive Hospital Intelligence Platform",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "calibrator_loaded": predictor.calibrator is not None,
        "model_version": predictor.model_version,
        "artifact_id": predictor.artifact_id,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)

def predict_readmission(patient: PatientFeatures):
    """
    Estimate calibrated 30-day readmission risk.
    """

    try:
        patient_data = patient.model_dump()

        patient_df = pd.DataFrame([patient_data])

        result = predictor.predict(patient_df)

        prediction_id = save_prediction(
            patient_reference="DEMO-PATIENT",
            raw_probability=float(result["raw_probability"]),
            calibrated_probability=float(result["calibrated_probability"]),
            risk_tier=result["risk_tier"],
            model_name=predictor.model_name,
            model_version=predictor.db_model_version,
        )


        return PredictionResponse(
            raw_probability=result["raw_probability"],
            calibrated_probability=result["calibrated_probability"],
            risk_tier=result["risk_tier"],
            prediction_id=prediction_id,
            model_version=predictor.model_version,
            artifact_id=predictor.artifact_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be generated.",
        ) from exc


@app.post("/explain")
def explain_readmission(patient: PatientFeatures):
    """
    Explain the XGBoost model's prediction for a patient.

    Returns the ten most influential transformed features.
    """
    try:
        patient_data = patient.model_dump()
        patient_df = pd.DataFrame([patient_data])

        explainer = _get_explainer()

        contributions = explainer.explain(
            patient_df,
            top_n=10
        )

        return {
            "status": "success",
            "explanation_type": "SHAP",
            "model": "readmission_xgb",
            "model_version": contributions["model_version"],
            "artifact_id": contributions["artifact_id"],
            "explanation_scale": contributions["explanation_scale"],
            "base_value": contributions["base_value"],
            "margin": contributions["margin"],
            "features": contributions["top_features"],
        }

    except Exception as exc:
        logger.exception("SHAP explanation failed")

        raise HTTPException(
            status_code=500,
            detail="SHAP explanation could not be generated."
        ) from exc


@app.get("/predictions")
def list_predictions(limit: int = 100):
    """
    Retrieve saved prediction history from PostgreSQL.
    """

    try:
        history = get_prediction_history(limit=limit)

        return {
            "total": len(history),
            "predictions": history
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history.",
        ) from exc

@app.get("/predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int):

    try:
        history = get_prediction_history(limit=100)

        for row in history:
            if row["id"] == prediction_id:
                return row

        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error retrieving prediction: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction."
        )
