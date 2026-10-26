"""Operational risk policy for 30-day readmission predictions."""

LOW_RISK_THRESHOLD = 0.10
HIGH_RISK_THRESHOLD = 0.15
INTERVENTION_THRESHOLD = 0.15


def assign_risk_tier(probability: float) -> str:
    """Convert a calibrated readmission probability into a risk tier."""

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    if probability < HIGH_RISK_THRESHOLD:
        return "Moderate"

    return "High"


def requires_intervention(probability: float) -> bool:
    """Return whether the patient meets the intervention threshold."""

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    return probability >= INTERVENTION_THRESHOLD
