"""Data handling for the prediction history and overview pages."""

import pandas as pd

HISTORY_COLUMNS = [
    "id",
    "patient_reference",
    "calibrated_probability",
    "risk_tier",
    "model_version",
    "created_at",
]

RISK_ORDER = ["Low", "Moderate", "High"]


def prepare_history(payload):
    """Convert a GET /predictions response into a clean DataFrame.

    Returns an empty frame (with the expected columns) when there are no
    predictions. Rows with an unusable probability or timestamp are dropped.
    Raises ValueError if the payload structure is invalid.
    """
    if not isinstance(payload, dict):
        raise ValueError("History response is not a JSON object.")

    rows = payload.get("predictions", [])
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("'predictions' must be a list.")

    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.DataFrame(rows)
    missing = set(HISTORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"History response is missing: {', '.join(sorted(missing))}"
        )

    df = df[HISTORY_COLUMNS].copy()
    df["calibrated_probability"] = pd.to_numeric(
        df["calibrated_probability"], errors="coerce"
    )
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["calibrated_probability", "created_at"])
    df["risk_tier"] = df["risk_tier"].astype(str)
    return df.reset_index(drop=True)


def filter_history(df, risk_tiers=None, start_date=None, end_date=None):
    """Filter by risk tiers and an inclusive date range."""
    out = df
    if risk_tiers:
        out = out[out["risk_tier"].isin(risk_tiers)]
    if start_date is not None:
        out = out[out["created_at"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        out = out[out["created_at"] < end]
    return out


SORT_OPTIONS = {
    "Newest first": ("created_at", False),
    "Oldest first": ("created_at", True),
    "Highest risk": ("calibrated_probability", False),
    "Lowest risk": ("calibrated_probability", True),
    "Prediction ID": ("id", True),
}


def sort_history(df, option):
    column, ascending = SORT_OPTIONS[option]
    return df.sort_values(column, ascending=ascending).reset_index(drop=True)


def display_table(df):
    """Rename columns for presentation."""
    return df.rename(
        columns={
            "id": "Prediction ID",
            "patient_reference": "Patient reference",
            "calibrated_probability": "Calibrated probability",
            "risk_tier": "Risk category",
            "model_version": "Model version",
            "created_at": "Created at",
        }
    )


def to_csv_bytes(df):
    return display_table(df).to_csv(index=False).encode("utf-8")


def summarize(df):
    """Headline metrics for the overview page."""
    if df.empty:
        return {"total": 0, "average_risk": None, "high_risk_share": None}
    return {
        "total": int(len(df)),
        "average_risk": float(df["calibrated_probability"].mean()),
        "high_risk_share": float((df["risk_tier"] == "High").mean()),
    }


def risk_distribution(df):
    counts = df["risk_tier"].value_counts()
    ordered = [t for t in RISK_ORDER if t in counts.index]
    ordered += [t for t in counts.index if t not in RISK_ORDER]
    return (
        counts.reindex(ordered)
        .rename_axis("risk_tier")
        .reset_index(name="count")
    )


def daily_activity(df):
    """Predictions per day and mean calibrated risk per day."""
    if df.empty:
        return pd.DataFrame(columns=["date", "count", "mean_risk"])
    daily = (
        df.assign(date=df["created_at"].dt.normalize())
        .groupby("date")
        .agg(count=("id", "count"), mean_risk=("calibrated_probability", "mean"))
        .reset_index()
    )
    return daily


RISK_COLORS = {"Low": "#2CA58D", "Moderate": "#F2B134", "High": "#EF553B"}


def _style(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_distribution_figure(df):
    import plotly.express as px

    dist = risk_distribution(df)
    fig = px.bar(
        dist,
        x="risk_tier",
        y="count",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        labels={"risk_tier": "Risk category", "count": "Predictions"},
        title="Risk category distribution",
    )
    fig.update_layout(showlegend=False)
    return _style(fig)


def activity_figure(df):
    import plotly.graph_objects as go

    daily = daily_activity(df)
    fig = go.Figure()
    fig.add_bar(
        x=daily["date"], y=daily["count"], name="Predictions",
        marker_color="#4C9BE8",
    )
    fig.add_scatter(
        x=daily["date"], y=daily["mean_risk"], name="Mean calibrated risk",
        mode="lines+markers", yaxis="y2", line=dict(color="#F2B134"),
    )
    fig.update_layout(
        title="Prediction activity over time",
        yaxis=dict(title="Predictions per day"),
        yaxis2=dict(
            title="Mean calibrated risk", overlaying="y", side="right",
            tickformat=".0%", range=[0, 1],
        ),
        legend=dict(orientation="h", y=-0.2),
    )
    return _style(fig)
