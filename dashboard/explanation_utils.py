"""Helpers for presenting SHAP explanations returned by POST /explain."""

import re

import pandas as pd

POSITIVE_COLOR = "#EF553B"
NEGATIVE_COLOR = "#4C9BE8"

_PREFIX_LABELS = {
    "num": "",
    "numeric": "",
    "cat": "",
    "categorical": "",
    "remainder": "",
}

_NAME_OVERRIDES = {
    "age_midpoint": "Age (midpoint)",
    "length_of_stay": "Length of stay",
    "hba1c_result": "HbA1c result",
    "lab_procedure_count": "Laboratory procedures",
    "procedure_count": "Other procedures",
    "medication_count": "Number of medications",
    "diagnosis_count": "Number of diagnoses",
    "previous_inpatient_visits": "Previous inpatient visits",
    "previous_outpatient_visits": "Previous outpatient visits",
    "previous_emergency_visits": "Previous emergency visits",
    "total_previous_visits": "Total previous visits",
    "has_previous_inpatient_visit": "Has previous inpatient visit",
}

_KNOWN_COLUMNS = sorted(
    [
        "age_midpoint", "gender", "race", "admission_type",
        "admission_source", "discharge_disposition", "length_of_stay",
        "previous_inpatient_visits", "previous_outpatient_visits",
        "previous_emergency_visits", "diagnosis_count",
        "lab_procedure_count", "procedure_count", "medication_count",
        "hba1c_result", "glucose_result", "insulin_status",
        "diabetes_medication", "primary_diagnosis_category",
        "secondary_diagnosis_category", "tertiary_diagnosis_category",
        "total_previous_visits", "has_previous_inpatient_visit",
    ],
    key=len,
    reverse=True,
)


def _humanize(text):
    text = _NAME_OVERRIDES.get(text, text)
    text = re.sub(r"[_\s]+", " ", str(text)).strip()
    return text[:1].upper() + text[1:] if text else text


def readable_feature_name(name):
    """Turn e.g. 'cat__primary_diagnosis_category_Circulatory' into
    'Primary diagnosis category: Circulatory'. Display only."""
    name = str(name)
    if "__" in name:
        prefix, rest = name.split("__", 1)
        if prefix in _PREFIX_LABELS:
            name = rest

    for column in _KNOWN_COLUMNS:
        if name == column:
            return _humanize(column)
        if name.startswith(column + "_"):
            category = name[len(column) + 1:]
            return f"{_humanize(column)}: {category}"
    return _humanize(name)


def parse_explanation(payload):
    """Validate an /explain response and return a display DataFrame.

    Raises ValueError if the payload is malformed.
    """
    if not isinstance(payload, dict):
        raise ValueError("Explanation response is not a JSON object.")

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Explanation response contains no features.")

    df = pd.DataFrame(features)
    missing = {"feature", "shap_value", "feature_value"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Explanation response is missing: {', '.join(sorted(missing))}"
        )

    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="raise")
    df["feature_value"] = pd.to_numeric(
        df["feature_value"], errors="coerce"
    )
    df["label"] = df["feature"].map(readable_feature_name)
    df["direction"] = df["shap_value"].apply(
        lambda v: "Increases score" if v > 0 else "Decreases score"
    )
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).head(10).reset_index(
        drop=True
    )


def build_shap_figure(df):
    """Horizontal Plotly bar chart, largest impact at the top."""
    import plotly.graph_objects as go

    plot_df = df.iloc[::-1]
    colors = [
        POSITIVE_COLOR if v > 0 else NEGATIVE_COLOR
        for v in plot_df["shap_value"]
    ]
    fig = go.Figure(
        go.Bar(
            x=plot_df["shap_value"],
            y=plot_df["label"],
            orientation="h",
            marker_color=colors,
            customdata=plot_df[["feature", "feature_value"]],
            hovertemplate=(
                "<b>%{y}</b><br>SHAP value: %{x:.4f}"
                "<br>Transformed value: %{customdata[1]}"
                "<br>Raw name: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(320, 40 * len(plot_df) + 100),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="SHAP value (effect on XGBoost raw score)",
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0, line_width=1, line_color="#888")
    return fig
