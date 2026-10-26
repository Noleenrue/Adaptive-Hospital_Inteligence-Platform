import pytest

from src.models.risk_policy import (
    assign_risk_tier,
    requires_intervention,
)


def test_low_risk():
    assert assign_risk_tier(0.05) == "Low"


def test_moderate_risk():
    assert assign_risk_tier(0.12) == "Moderate"


def test_high_risk():
    assert assign_risk_tier(0.20) == "High"


def test_boundary_at_010():
    assert assign_risk_tier(0.10) == "Moderate"


def test_boundary_at_015():
    assert assign_risk_tier(0.15) == "High"


def test_intervention():
    assert requires_intervention(0.15) is True
    assert requires_intervention(0.14) is False


def test_invalid_probability():
    with pytest.raises(ValueError):
        assign_risk_tier(1.2)
