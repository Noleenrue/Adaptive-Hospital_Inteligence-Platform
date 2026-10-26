
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.predictor import ReadmissionPredictor

# Contributions must reconstruct the deployed margin to this tolerance.
MARGIN_TOLERANCE = 1e-3


class ReadmissionSHAPExplainer:
    """
    Explain XGBoost readmission predictions with exact TreeSHAP values.

    Contributions come from the deployed booster itself, evaluated on the
    exact matrix the pipeline feeds to ``predict`` (same inputs, feature
    order and sparse missing-value semantics). They are on the raw margin
    (log-odds) scale, not the calibrated probability, and sum with the
    base value to the deployed raw margin.

    The previous implementation densified the matrix first, which turned
    XGBoost's "missing" entries into real zeros and produced explanations
    for a different model input.
    """

    def __init__(self, predictor=None):
        self.predictor = predictor or ReadmissionPredictor()

        self.preprocessor = self.predictor.model.named_steps[
            "preprocessor"
        ]

        self.classifier = self.predictor.model.named_steps[
            "classifier"
        ]

        self.booster = self.classifier.get_booster()

    def contributions(self, patient_features):
        """
        Return (table, base_value, margin) for exactly one patient.

        ``table`` has every transformed feature in pipeline order with its
        value and contribution; ``margin`` is the deployed raw margin.
        """

        if isinstance(patient_features, dict):
            patient_features = pd.DataFrame(
                [patient_features]
            )

        if not isinstance(patient_features, pd.DataFrame):
            raise TypeError(
                "Expected a dictionary or pandas DataFrame."
            )

        if len(patient_features) != 1:
            raise ValueError(
                "Explain exactly one patient at a time."
            )

        transformed = self.preprocessor.transform(
            self.predictor.prepare_features(patient_features)
        )

        feature_names = (
            self.preprocessor.get_feature_names_out()
        )

        # Native TreeSHAP on the exact pipeline output; the matrix is not
        # densified so XGBoost keeps its sparse missing-value semantics.
        values = self.booster.predict(
            xgb.DMatrix(transformed),
            pred_contribs=True,
        )

        if values.shape != (1, len(feature_names) + 1):
            raise ValueError(
                "Unexpected SHAP output shape."
            )

        margin = float(
            self.classifier.predict(transformed, output_margin=True)[0]
        )
        base_value = float(values[0, -1])

        if abs(values[0].sum() - margin) > MARGIN_TOLERANCE:
            raise ValueError(
                "SHAP contributions do not reconstruct the model margin."
            )

        dense = (
            transformed.toarray()
            if hasattr(transformed, "toarray")
            else np.asarray(transformed)
        )

        table = pd.DataFrame({
            "feature": feature_names,
            "shap_value": values[0, :-1],
            "feature_value": dense[0],
        })

        return table, base_value, margin

    def explain(self, patient_features, top_n=10):

        table, base_value, margin = self.contributions(patient_features)

        table["absolute_impact"] = table["shap_value"].abs()

        top = table.sort_values(
            "absolute_impact",
            ascending=False
        ).head(top_n)

        return {
            "explanation_scale": "XGBoost raw model score",
            "base_value": base_value,
            "margin": margin,
            "raw_probability": float(1.0 / (1.0 + np.exp(-margin))),
            "model_version": self.predictor.model_version,
            "artifact_id": self.predictor.artifact_id,
            "top_features": top[
                ["feature", "feature_value", "shap_value"]
            ].to_dict(orient="records")
        }
