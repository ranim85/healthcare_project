import joblib
import pandas as pd
import pytest

from config.config import MODEL_PATH


def test_pipeline_handles_nan_at_inference_if_artifact_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found. Run training first.")

    pipeline = joblib.load(MODEL_PATH)
    patient = pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [None],
            "BloodPressure": [80.0],
            "SkinThickness": [None],
            "Insulin": [100.0],
            "BMI": [31.0],
            "DiabetesPedigreeFunction": [0.45],
            "Age": [42],
        }
    )

    proba = pipeline.predict_proba(patient)[0, 1]
    assert 0 <= proba <= 1


def test_pipeline_handles_clinical_extreme_values_if_artifact_exists():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found. Run training first.")

    pipeline = joblib.load(MODEL_PATH)
    patient = pd.DataFrame(
        {
            "Pregnancies": [20],
            "Glucose": [500.0],
            "BloodPressure": [250.0],
            "SkinThickness": [120.0],
            "Insulin": [900.0],
            "BMI": [80.0],
            "DiabetesPedigreeFunction": [3.0],
            "Age": [120],
        }
    )

    proba = pipeline.predict_proba(patient)[0, 1]
    assert 0 <= proba <= 1
