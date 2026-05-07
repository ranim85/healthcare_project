import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from config.config import FEATURES, MODEL_PATH, SHAP_BACKGROUND_PATH
from src.evaluation.xai import get_patient_explanation
from src.utils.logger import sys_logger as logger
from src.utils.sklearn import configure_sklearn_output

configure_sklearn_output()
st.set_page_config(page_title="Clinical Risk Predictor", layout="wide")
st.title("Diabetes Risk Assessment")


@st.cache_resource
def load_system():
    try:
        pipeline = joblib.load(MODEL_PATH)
        background = joblib.load(SHAP_BACKGROUND_PATH)
        return pipeline, background
    except Exception as e:
        logger.error(f"UI Loading Error: {e}")
        return None, None


pipeline, background = load_system()

if pipeline is None:
    st.error("Model artifacts not found. Please run: python -m src.models.train")
    st.stop()

with st.sidebar:
    st.header("Patient Biometrics")
    inputs = {}
    for feat in FEATURES:
        if feat == "Pregnancies":
            inputs[feat] = st.slider(feat, 0, 20, 3)
        elif feat == "Age":
            inputs[feat] = st.slider(feat, 0, 120, 45)
        elif feat == "DiabetesPedigreeFunction":
            inputs[feat] = st.slider(feat, 0.0, 2.5, 0.5)
        else:
            inputs[feat] = st.number_input(f"{feat}", value=100.0)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Patient Summary")
    input_df = pd.DataFrame([inputs])
    st.table(input_df.T.rename(columns={0: "Value"}))

    risk_proba = pipeline.predict_proba(input_df)[0, 1]

    st.markdown(f"### Predicted Risk: **{risk_proba:.1%}**")
    if risk_proba > 0.5:
        st.error("High Risk Level")
    else:
        st.success("Normal Risk Level")

with col2:
    st.subheader("Explainable AI (SHAP)")
    try:
        explanation = get_patient_explanation(pipeline, input_df, background)
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.info("Explanation currently unavailable.")

        logger.warning(f"SHAP UI Error: {e}")


st.divider()
st.caption("Local ML engineering project. Not for clinical use.")
