"""
app.py — Streamlit Deployment
==============================
Run locally:
    cd streamlit_app
    streamlit run app.py

The app loads pre-trained joblib models and lets users enter clinical /
lifestyle features to get predictions for all three tasks.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes ML Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ══════════════════════════════════════════════════════════════════════════
# Helper: load model / scaler safely
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifact(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — Feature Inputs
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render all feature sliders/selects in the sidebar and return values."""
    st.sidebar.header("🔧 Patient Feature Inputs")
    st.sidebar.markdown("Adjust the sliders to match the patient's profile.")

    features = {}

    with st.sidebar.expander("🏥 Clinical Indicators", expanded=True):
        features["age"]            = st.slider("Age (years)",        18, 100, 45)
        features["bmi"]            = st.slider("BMI",                10.0, 60.0, 27.5, 0.5)
        features["blood_glucose"]  = st.slider("Fasting Glucose (mg/dL)", 50, 400, 100)
        features["hba1c"]          = st.slider("HbA1c (%)",          4.0, 15.0, 5.7, 0.1)
        features["systolic_bp"]    = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        features["diastolic_bp"]   = st.slider("Diastolic BP (mmHg)",40, 130, 80)

    with st.sidebar.expander("🍎 Lifestyle Indicators", expanded=True):
        features["smoking"]        = st.selectbox("Smoking Status",
            ["Never", "Former", "Current"], index=0)
        features["alcohol"]        = st.selectbox("Alcohol Use",
            ["None", "Moderate", "Heavy"], index=0)
        features["physical_activity"] = st.slider("Physical Activity (days/week)", 0, 7, 3)
        features["diet_quality"]   = st.slider("Diet Quality Score (0–10)", 0, 10, 5)
        features["sleep_hours"]    = st.slider("Average Sleep (hours/night)", 3, 12, 7)

    with st.sidebar.expander("🧬 Diagnostic & History", expanded=False):
        features["family_history"] = st.selectbox("Family History of Diabetes",
            ["No", "Yes"], index=0)
        features["hypertension"]   = st.selectbox("Hypertension",
            ["No", "Yes"], index=0)
        features["cholesterol"]    = st.slider("Total Cholesterol (mg/dL)", 100, 400, 200)
        features["triglycerides"]  = st.slider("Triglycerides (mg/dL)",     50, 600, 150)
        features["waist_cm"]       = st.slider("Waist Circumference (cm)",  50, 160, 90)

    return features


# ══════════════════════════════════════════════════════════════════════════
# Feature Vector Encoding
# ══════════════════════════════════════════════════════════════════════════

SMOKE_MAP   = {"Never": 0, "Former": 1, "Current": 2}
ALCOHOL_MAP = {"None": 0, "Moderate": 1, "Heavy": 2}
YESNO_MAP   = {"No": 0, "Yes": 1}

def encode_features(feat: dict, feature_cols: list | None) -> pd.DataFrame:
    """
    Convert the raw sidebar inputs to a numeric DataFrame row.
    If feature_cols is supplied (from training), align to that column order.
    """
    row = {
        "age"              : feat["age"],
        "bmi"              : feat["bmi"],
        "blood_glucose"    : feat["blood_glucose"],
        "hba1c"            : feat["hba1c"],
        "systolic_bp"      : feat["systolic_bp"],
        "diastolic_bp"     : feat["diastolic_bp"],
        "smoking"          : SMOKE_MAP[feat["smoking"]],
        "alcohol"          : ALCOHOL_MAP[feat["alcohol"]],
        "physical_activity": feat["physical_activity"],
        "diet_quality"     : feat["diet_quality"],
        "sleep_hours"      : feat["sleep_hours"],
        "family_history"   : YESNO_MAP[feat["family_history"]],
        "hypertension"     : YESNO_MAP[feat["hypertension"]],
        "cholesterol"      : feat["cholesterol"],
        "triglycerides"    : feat["triglycerides"],
        "waist_cm"         : feat["waist_cm"],
    }
    df_row = pd.DataFrame([row])

    if feature_cols is not None:
        # Add missing columns (if any) with 0 and reorder
        for col in feature_cols:
            if col not in df_row.columns:
                df_row[col] = 0
        df_row = df_row[feature_cols]

    return df_row


# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Binary Classification
# ══════════════════════════════════════════════════════════════════════════

def tab_binary(features: dict, feature_cols) -> None:
    st.header("🔵 Binary Classification — Diabetes Diagnosis")
    st.markdown(
        "Predicts whether the patient is likely **diabetic (Yes)** "
        "or **not (No)** based on their health indicators."
    )

    model  = load_artifact("binary_log_reg_model.joblib")
    scaler = load_artifact("binary_log_reg_model.joblib")

    if model is None:
        st.warning("⚠️ Binary model not found. Please run the notebook to train and export models first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔍 Predict Diabetes Diagnosis", use_container_width=True,
                     type="primary"):
            X_row = encode_features(features, feature_cols)
            if scaler is not None:
                X_row_sc = pd.DataFrame(
                    scaler.transform(X_row), columns=X_row.columns)
            else:
                X_row_sc = X_row

            prediction = model.predict(X_row_sc)[0]
            probability = model.predict_proba(X_row_sc)[0]

            st.markdown("---")
            if prediction == 1:
                st.error("### 🔴 Prediction: **Diabetic — Yes**")
            else:
                st.success("### 🟢 Prediction: **Diabetic — No**")

            conf_neg, conf_pos = probability[0], probability[1]
            st.metric("Confidence (No Diabetes)",  f"{conf_neg*100:.1f}%")
            st.metric("Confidence (Diabetes)",     f"{conf_pos*100:.1f}%")

            # Simple risk bar
            st.markdown("**Risk Level:**")
            st.progress(int(conf_pos * 100))
            st.caption(f"Risk score: {conf_pos*100:.1f} / 100")

    with col2:
        st.info(
            "**How to interpret**\n\n"
            "- **No** → Low probability of diabetes.\n"
            "- **Yes** → High probability; further clinical tests recommended.\n\n"
            "Confidence bars show the model's certainty."
        )


# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Multiclass Classification
# ══════════════════════════════════════════════════════════════════════════

STAGE_LABELS = {
    0: "✅ No Diabetes",
    1: "⚠️ Pre-Diabetes",
    2: "🔴 Type 2 Diabetes",
    3: "🟣 Type 1 Diabetes",
}

def tab_multiclass(features: dict, feature_cols) -> None:
    st.header("🟡 Multiclass Classification — Diabetes Stage")
    st.markdown(
        "Predicts the **stage of diabetes**: No Diabetes, Pre-Diabetes, "
        "Type 2 Diabetes, or Type 1 Diabetes."
    )

    model  = load_artifact("multiclass_model.joblib")
    scaler = load_artifact("multiclass_scaler.joblib")

    if model is None:
        st.warning("⚠️ Multiclass model not found. Please run the notebook first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔍 Predict Diabetes Stage", use_container_width=True,
                     type="primary"):
            X_row = encode_features(features, feature_cols)
            if scaler is not None:
                X_row_sc = pd.DataFrame(
                    scaler.transform(X_row), columns=X_row.columns)
            else:
                X_row_sc = X_row

            prediction   = model.predict(X_row_sc)[0]
            probabilities = model.predict_proba(X_row_sc)[0]

            stage_label = STAGE_LABELS.get(int(prediction), f"Stage {prediction}")
            st.markdown("---")
            st.markdown(f"### Predicted Stage: **{stage_label}**")

            st.markdown("**Class Probabilities:**")
            prob_df = pd.DataFrame({
                "Stage"      : [STAGE_LABELS.get(i, f"Stage {i}")
                                for i in range(len(probabilities))],
                "Probability": [f"{p*100:.1f}%" for p in probabilities],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with col2:
        st.info(
            "**Stage descriptions**\n\n"
            "- **No Diabetes** → Normal glucose levels.\n"
            "- **Pre-Diabetes** → Borderline; lifestyle change can reverse.\n"
            "- **Type 2** → Insulin resistance; most common form.\n"
            "- **Type 1** → Autoimmune; requires insulin therapy."
        )


# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Regression
# ══════════════════════════════════════════════════════════════════════════

def tab_regression(features: dict, feature_cols) -> None:
    st.header("🟢 Regression — Diabetes Risk Score")
    st.markdown(
        "Predicts a **continuous risk score** (diabetes_risk_score) "
        "representing the patient's overall diabetes risk."
    )

    model  = load_artifact("regression_model.joblib")
    scaler = load_artifact("regression_scaler.joblib")

    if model is None:
        st.warning("⚠️ Regression model not found. Please run the notebook first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔍 Predict Risk Score", use_container_width=True,
                     type="primary"):
            X_row = encode_features(features, feature_cols)
            if scaler is not None:
                X_row_sc = pd.DataFrame(
                    scaler.transform(X_row), columns=X_row.columns)
            else:
                X_row_sc = X_row

            prediction = model.predict(X_row_sc)[0]
            st.markdown("---")
            st.metric("Predicted Diabetes Risk Score", f"{prediction:.2f}")

            # Risk interpretation
            if prediction < 30:
                st.success("🟢 Low Risk — Maintain current healthy habits.")
            elif prediction < 60:
                st.warning("🟡 Moderate Risk — Consider lifestyle adjustments.")
            else:
                st.error("🔴 High Risk — Consult a healthcare professional.")

            st.progress(min(int(prediction), 100))
            st.caption("Score range: 0 (no risk) → 100 (maximum risk)")

    with col2:
        st.info(
            "**Score ranges**\n\n"
            "- **0 – 30** → Low risk\n"
            "- **30 – 60** → Moderate risk\n"
            "- **60 – 100** → High risk\n\n"
            "This is a model estimate and not a clinical diagnosis."
        )


# ══════════════════════════════════════════════════════════════════════════
# Main App Layout
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Title ──────────────────────────────────────────────────────────────
    st.title("🩺 Diabetes Health Indicators — ML Predictor")
    st.markdown(
        "An interactive machine learning application for predicting diabetes "
        "diagnosis, stage classification, and risk score from clinical and "
        "lifestyle features."
    )
    st.markdown("---")

    # ── Load shared artifacts ──────────────────────────────────────────────
    feature_cols = load_artifact("feature_columns.joblib")

    # ── Sidebar ────────────────────────────────────────────────────────────
    features = render_sidebar()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Binary Classification",
        "🟡 Multiclass Classification",
        "🟢 Regression",
        "📖 About",
    ])

    with tab1:
        tab_binary(features, feature_cols)

    with tab2:
        tab_multiclass(features, feature_cols)

    with tab3:
        tab_regression(features, feature_cols)

    with tab4:
        st.header("📖 About This Project")
        st.markdown(
            """
### Diabetes Health Indicators ML Project

This application was built as part of a machine learning project that
uses the **Diabetes Health Indicators Dataset** to predict health outcomes.

#### 🗂 Three Prediction Tasks
| Task | Target Column | Type |
|------|--------------|------|
| Binary Classification | `diagnosed_diabetes` | Yes / No |
| Multiclass Classification | `diabetes_stage` | 4 stages |
| Regression | `diabetes_risk_score` | Continuous (0–100) |

#### ⚙️ Models Used
- **Binary:** Logistic Regression, Decision Tree, KNN (tuned via Grid Search)
- **Multiclass:** Decision Tree, Logistic Regression, KNN (tuned via Random Search)
- **Regression:** Linear Regression, Decision Tree Regressor (tuned via Grid Search)

#### 🛠 Tech Stack
`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn` · `Streamlit` · `joblib`

#### ⚠️ Disclaimer
This tool is for educational purposes only and is **not** a substitute
for professional medical advice, diagnosis, or treatment.
            """
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Built with ❤️ using Python & Streamlit · "
        "Diabetes Health Indicators ML Project · AtomCamp"
    )


if __name__ == "__main__":
    main()
