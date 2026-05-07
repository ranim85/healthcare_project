import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.models import api
from src.models.api import PatientData, _safe_model_path, app


def test_patient_payload_contract():
    payload = PatientData(
        Pregnancies=1,
        Glucose=110.0,
        BloodPressure=70.0,
        SkinThickness=20.0,
        Insulin=75.0,
        BMI=27.5,
        DiabetesPedigreeFunction=0.4,
        Age=35,
    )

    assert payload.Age == 35


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_rejects_corrupted_input():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "Pregnancies": 1,
            "Glucose": 9999,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 75,
            "BMI": 27.5,
            "DiabetesPedigreeFunction": 0.4,
            "Age": 35,
            "unexpected": "field",
        },
    )

    assert response.status_code == 422


def test_api_predict_smoke_with_stubbed_pipeline(monkeypatch):
    class StubPipeline:
        def predict_proba(self, _input_df):
            return np.array([[0.2, 0.8]])

        def predict(self, _input_df):
            return np.array([1])

    monkeypatch.setattr(api, "pipeline", StubPipeline())
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "Pregnancies": 1,
            "Glucose": 120.0,
            "BloodPressure": 70.0,
            "SkinThickness": 20.0,
            "Insulin": 75.0,
            "BMI": 27.5,
            "DiabetesPedigreeFunction": 0.4,
            "Age": 35,
        },
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_model_path_traversal_is_rejected():
    with pytest.raises(ValueError):
        _safe_model_path(api.MODELS_DIR.parent / "untrusted.joblib")
