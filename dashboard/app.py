
import streamlit as st
import pandas as pd

from api_client import (
    DashboardAPIError,
    get_health,
    get_history,
    post_explanation,
    post_prediction,
    validate_patient_data,
)

from explanation_utils import (
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    build_shap_figure,
    parse_explanation,
)

from history_utils import (
    SORT_OPTIONS,
    activity_figure,
    display_table,
    filter_history,
    prepare_history,
    risk_distribution_figure,
    sort_history,
    summarize,
    to_csv_bytes,
)

st.set_page_config(
    page_title="Adaptive Hospital Intelligence Platform",
    page_icon="🏥",
    layout="wide"
)


def load_history():
    """Fetch saved predictions; show an error and return None on failure."""
    try:
        return prepare_history(get_history())
    except DashboardAPIError as exc:
        st.error(str(exc))
    except ValueError:
        st.error(
            "The API returned invalid prediction history data. "
            "Contact the platform administrator."
        )
    return None


st.title("🏥 Adaptive Hospital Intelligence Platform")
st.warning(
    "Research prototype — not clinically validated. It estimates 30-day "
    "readmission risk at hospital discharge, not at admission. Risk "
    "categories are not validated clinical decision rules; Low risk does "
    "not mean zero risk."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Patient Risk Assessment",
        "Prediction History",
    ],
)

if page == "Overview":
    st.header("Platform Overview")

    try:
        health = get_health()

        st.success("FastAPI backend is connected")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model Loaded", "Yes")

        with col2:
            st.metric("Calibrator Loaded", "Yes")

        with col3:
            st.metric("API-reported model version", health["model_version"])

        with col4:
            st.metric("API-reported artifact ID", health["artifact_id"])

        st.caption(
            "Identity reported by the configured API; it is not hardcoded "
            "in the dashboard."
        )

    except DashboardAPIError as exc:
        st.error(str(exc))

    st.subheader("Prediction Summary")

    history = load_history()

    if history is not None and history.empty:
        st.info("No predictions saved yet. Run a patient assessment first.")
    elif history is not None:
        stats = summarize(history)
        m1, m2, m3 = st.columns(3)
        m1.metric("Saved Predictions", stats["total"])
        m2.metric("Average Calibrated Risk", f"{stats['average_risk']:.1%}")
        m3.metric("High-Risk Share", f"{stats['high_risk_share']:.1%}")

        c1, c2 = st.columns(2)
        c1.plotly_chart(
            risk_distribution_figure(history), use_container_width=True
        )
        c2.plotly_chart(activity_figure(history), use_container_width=True)
        st.caption("Based on the 100 most recent saved predictions.")


elif page == "Patient Risk Assessment":
    st.header("Patient Risk Assessment")
    st.info(
        "This assessment is intended only at hospital discharge, when "
        "discharge disposition, length of stay, and completed encounter "
        "information are available. It is not an admission-time prediction. "
        "Research risk categories are not validated clinical decision rules; "
        "Low does not mean zero risk. This model is not clinically validated."
    )

    st.caption(
        "A successful submission creates a saved prediction record. The "
        "button is locked after a submission attempt until you explicitly "
        "start a new assessment. This UI-only safeguard does not guarantee "
        "exactly-once writes, especially if a request times out."
    )

    if st.session_state.get("prediction_submission_locked"):
        if st.button("Start a new assessment", key="start_new_assessment"):
            st.session_state.pop("assessment", None)
            st.session_state.pop("prediction_submission_locked", None)
            st.rerun()

    with st.form("patient_risk_form"):
        st.subheader("1. Patient Demographics")

        age_midpoint = st.number_input(
            "Age (midpoint of age range)",
            min_value=0.0,
            max_value=120.0,
            value=65.0,
        )
        gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
        race = st.text_input("Race category", value="Caucasian")

        st.subheader("2. Hospital Admission")
        admission_type = st.text_input("Admission type", value="Emergency")
        admission_source = st.text_input(
            "Admission source", value="Emergency Room"
        )
        discharge_disposition = st.text_input(
            "Discharge disposition", value="Discharged to home"
        )
        length_of_stay = st.number_input(
            "Length of stay (days)", min_value=0, value=4
        )

        st.subheader("3. Previous Hospital Visits")
        previous_inpatient = st.number_input(
            "Previous inpatient visits", min_value=0, value=1
        )
        previous_outpatient = st.number_input(
            "Previous outpatient visits", min_value=0, value=0
        )
        previous_emergency = st.number_input(
            "Previous emergency visits", min_value=0, value=0
        )

        st.subheader("4. Clinical Tests and Medications")
        diagnosis_count = st.number_input(
            "Number of diagnoses", min_value=0, value=5
        )
        lab_count = st.number_input(
            "Laboratory procedures", min_value=0, value=40
        )
        procedure_count = st.number_input(
            "Other procedures", min_value=0, value=1
        )
        medication_count = st.number_input(
            "Number of medications", min_value=0, value=12
        )

        hba1c_result = st.selectbox(
            "HbA1c result", ["None", "Norm", ">7", ">8"]
        )
        glucose_result = st.selectbox(
            "Glucose result", ["None", "Norm", ">200", ">300"]
        )
        insulin_status = st.selectbox(
            "Insulin status", ["No", "Steady", "Up", "Down"]
        )
        diabetes_medication = st.selectbox(
            "Diabetes medication", ["Yes", "No"]
        )

        st.subheader("5. Diagnosis Categories")
        st.caption(
            "Use the category labels produced by your model's "
            "diagnosis preprocessing pipeline."
        )
        primary_diagnosis = st.text_input(
            "Primary diagnosis category", value="Circulatory"
        )
        secondary_diagnosis = st.text_input(
            "Secondary diagnosis category", value="Diabetes"
        )
        tertiary_diagnosis = st.text_input(
            "Tertiary diagnosis category", value="Other"
        )

        submitted = st.form_submit_button(
            "Predict Readmission Risk",
            type="primary",
            disabled=st.session_state.get(
                "prediction_submission_locked", False
            ),
        )

    if submitted:
        patient_data = {
            "age_midpoint": age_midpoint,
            "gender": gender,
            "race": race,
            "admission_type": admission_type,
            "admission_source": admission_source,
            "discharge_disposition": discharge_disposition,
            "length_of_stay": length_of_stay,
            "previous_inpatient_visits": previous_inpatient,
            "previous_outpatient_visits": previous_outpatient,
            "previous_emergency_visits": previous_emergency,
            "diagnosis_count": diagnosis_count,
            "lab_procedure_count": lab_count,
            "procedure_count": procedure_count,
            "medication_count": medication_count,
            "hba1c_result": hba1c_result,
            "glucose_result": glucose_result,
            "insulin_status": insulin_status,
            "diabetes_medication": diabetes_medication,
            "primary_diagnosis_category": primary_diagnosis,
            "secondary_diagnosis_category": secondary_diagnosis,
            "tertiary_diagnosis_category": tertiary_diagnosis,
            "total_previous_visits": (
                previous_inpatient
                + previous_outpatient
                + previous_emergency
            ),
            "has_previous_inpatient_visit": int(previous_inpatient > 0),
        }

        validation_error = validate_patient_data(patient_data)
        if validation_error:
            st.error(validation_error)
        else:
            try:
                health = get_health()
                st.session_state["prediction_submission_locked"] = True
                with st.spinner("Calculating readmission risk..."):
                    result = post_prediction(patient_data)

                # Reuse the same payload; failures must not affect the prediction.
                explanation_df, explanation_error = None, None
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        explanation_df = parse_explanation(
                            post_explanation(patient_data)
                        )
                    except DashboardAPIError as exc:
                        explanation_error = str(exc)
                    except ValueError:
                        explanation_error = (
                            "The API returned an invalid explanation. "
                            "The prediction is unaffected."
                        )

                st.session_state["assessment"] = {
                    "result": result,
                    "explanation": explanation_df,
                    "explanation_error": explanation_error,
                    "health": health,
                }

            except DashboardAPIError as exc:
                st.error(str(exc))
                if st.session_state.get("prediction_submission_locked"):
                    st.warning(
                        "The request may have reached the API before the "
                        "error occurred. Check Prediction History before "
                        "starting another assessment; the dashboard cannot "
                        "guarantee exactly-once writes."
                    )
            except (KeyError, TypeError, ValueError):
                st.error(
                    "The API returned an invalid prediction response. "
                    "Check Prediction History before retrying because the "
                    "request may already have been saved."
                )
                if st.session_state.get("prediction_submission_locked"):
                    st.warning(
                        "The submission remains locked to reduce accidental "
                        "duplicate records. Start a new assessment only "
                        "after checking Prediction History."
                    )

    assessment = st.session_state.get("assessment")

    if assessment:
        result = assessment["result"]

        try:
            st.success("Prediction completed and saved by the API.")

            probability = float(result["calibrated_probability"])
            raw_probability = float(result["raw_probability"])
            risk_tier = result["risk_tier"]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Calibrated Risk", f"{probability:.1%}")
            col2.metric("Raw Model Probability", f"{raw_probability:.1%}")
            col3.metric("Risk Category", risk_tier)
            col4.metric(
                "Prediction ID",
                result.get("prediction_id", "N/A"),
            )

            st.progress(max(0.0, min(probability, 1.0)))

            with st.expander("Prediction Details"):
                st.json(result)

            st.caption(
                "The calibrated probability is the primary estimated risk; "
                "the raw model probability is shown separately for context. "
                "Low risk does not mean zero risk. These research categories "
                "are not validated clinical decision rules, and the model is "
                "not clinically validated."
            )
            if (
                result.get("model_version") != health.get("model_version")
                or result.get("artifact_id") != health.get("artifact_id")
            ):
                st.warning(
                    "The model identity in the prediction response differs "
                    "from the preceding health check. Do not rely on the "
                    "displayed identity; contact the platform administrator."
                )
            else:
                st.caption(
                    "Prediction model reported by API: "
                    f"{result.get('model_version')} "
                    f"({result.get('artifact_id')})."
                )

        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Unexpected prediction response: {exc}")

        st.divider()
        st.subheader("Why Did the Model Make This Prediction?")

        explanation_df = assessment["explanation"]

        if explanation_df is None:
            st.warning(
                "The explanation is currently unavailable. "
                "The prediction above is unaffected."
            )
            if assessment["explanation_error"]:
                st.caption(assessment["explanation_error"])
        else:
            st.plotly_chart(
                build_shap_figure(explanation_df),
                use_container_width=True,
            )

            st.markdown(
                f"<span style='color:{POSITIVE_COLOR}'>■</span> Positive "
                "SHAP value: increases the XGBoost raw prediction score "
                "&nbsp;&nbsp; "
                f"<span style='color:{NEGATIVE_COLOR}'>■</span> Negative "
                "SHAP value: decreases it",
                unsafe_allow_html=True,
            )

            st.caption(
                "SHAP values describe each feature's contribution to the "
                "XGBoost raw model score for this patient, relative to the "
                "model's baseline. They are not percentage changes in the "
                "calibrated risk shown above and do not imply causal "
                "effects."
            )

            table = explanation_df[
                ["label", "feature", "feature_value", "shap_value",
                 "direction"]
            ].rename(
                columns={
                    "label": "Feature",
                    "feature": "Model feature name",
                    "feature_value": "Transformed value",
                    "shap_value": "SHAP value",
                    "direction": "Effect on raw score",
                }
            )
            with st.expander("Detailed SHAP contributions", expanded=True):
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                )


elif page == "Prediction History":
    st.header("Prediction History")
    st.caption(
        "Saved predictions from PostgreSQL. The API returns up to the "
        "100 most recent records."
    )

    history = load_history()

    if history is None:
        st.stop()

    if history.empty:
        st.info("No predictions found.")
        st.stop()

    f1, f2, f3 = st.columns([2, 2, 1])
    tiers = sorted(history["risk_tier"].unique())
    selected_tiers = f1.multiselect("Risk category", tiers, default=tiers)

    min_d = history["created_at"].min().date()
    max_d = history["created_at"].max().date()
    date_range = f2.date_input(
        "Created between", value=(min_d, max_d),
        min_value=min_d, max_value=max_d,
    )
    sort_option = f3.selectbox("Sort by", list(SORT_OPTIONS))

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    filtered = sort_history(
        filter_history(history, selected_tiers, start, end), sort_option
    )

    st.metric("Predictions shown", f"{len(filtered)} of {len(history)}")

    if filtered.empty:
        st.info("No predictions match the selected filters.")
    else:
        st.dataframe(
            display_table(filtered),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Calibrated probability": st.column_config.ProgressColumn(
                    "Calibrated probability",
                    min_value=0.0, max_value=1.0, format="%.3f",
                ),
                "Created at": st.column_config.DatetimeColumn(
                    "Created at", format="YYYY-MM-DD HH:mm:ss"
                ),
            },
        )
        st.download_button(
            "Download CSV",
            data=to_csv_bytes(filtered),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
