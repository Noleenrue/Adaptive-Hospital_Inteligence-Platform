import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from history_utils import (  # noqa: E402
    HISTORY_COLUMNS,
    activity_figure,
    daily_activity,
    filter_history,
    prepare_history,
    risk_distribution,
    risk_distribution_figure,
    sort_history,
    summarize,
    to_csv_bytes,
)


def _row(i, prob, tier, ts):
    return {
        "id": i,
        "patient_reference": "DEMO-PATIENT",
        "raw_probability": 0.5,
        "calibrated_probability": prob,
        "risk_tier": tier,
        "model_name": "readmission_xgb",
        "model_version": "1.0",
        "created_at": ts,
    }


def _payload():
    return {
        "total": 4,
        "predictions": [
            _row(4, 0.80, "High", "2026-10-03T10:00:00"),
            _row(3, 0.40, "Moderate", "2026-10-02T09:00:00"),
            _row(2, 0.10, "Low", "2026-10-02T08:00:00"),
            _row(1, "bad", "Low", "2026-10-01T08:00:00"),
        ],
    }


def test_prepare_drops_invalid_rows():
    df = prepare_history(_payload())
    assert list(df.columns) == HISTORY_COLUMNS
    assert df["id"].tolist() == [4, 3, 2]


def test_prepare_empty_and_invalid():
    assert prepare_history({"predictions": []}).empty
    assert prepare_history({}).empty
    for bad in (None, [], {"predictions": "x"}, {"predictions": [{"id": 1}]}):
        try:
            prepare_history(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_filter_and_sort():
    df = prepare_history(_payload())
    out = filter_history(df, ["Low", "Moderate"], "2026-10-02", "2026-10-02")
    assert out["id"].tolist() == [3, 2]
    assert sort_history(df, "Highest risk")["id"].iloc[0] == 4
    assert sort_history(df, "Oldest first")["id"].iloc[0] == 2


def test_summary_and_distribution():
    df = prepare_history(_payload())
    s = summarize(df)
    assert s["total"] == 3
    assert abs(s["average_risk"] - (0.8 + 0.4 + 0.1) / 3) < 1e-9
    assert risk_distribution(df)["risk_tier"].tolist() == [
        "Low", "Moderate", "High"
    ]
    assert summarize(pd.DataFrame(columns=HISTORY_COLUMNS))["total"] == 0


def test_daily_activity_and_charts():
    df = prepare_history(_payload())
    daily = daily_activity(df)
    assert daily["count"].tolist() == [2, 1]
    assert activity_figure(df) is not None
    assert risk_distribution_figure(df) is not None


def test_csv_export():
    csv = to_csv_bytes(prepare_history(_payload())).decode()
    assert csv.splitlines()[0].startswith("Prediction ID,Patient reference")
    assert len(csv.splitlines()) == 4
