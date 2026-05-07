import joblib
import pandas as pd
import pytest

from config.config import MODEL_PATH
from src.models.pipeline import create_clinical_pipeline


def test_pipeline_creation():
    """Verifie la structure du pipeline Scikit-Learn."""
    pipeline = create_clinical_pipeline()
    steps = [step[0] for step in pipeline.steps]
    assert steps == ["imputer", "engineer", "scaler", "classifier"]


def test_inference_integrity():
    """Garantit que le modele charge produit des probabilites valides."""
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception:
        pytest.skip("Model artifact not found. Run training first.")

    test_patient = pd.DataFrame(
        {
            "Pregnancies": [3],
            "Glucose": [120],
            "BloodPressure": [80],
            "SkinThickness": [20],
            "Insulin": [100],
            "BMI": [30],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [40],
        }
    )

    proba = pipeline.predict_proba(test_patient)[0, 1]
    assert 0 <= proba <= 1
