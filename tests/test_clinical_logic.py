import pandas as pd
import pandera.errors as pa_errors
import pytest

from src.data.schemas import clinical_schema
from src.features.transformers import MedicalFeatureEngineer


@pytest.fixture
def sample_patient_data():
    return pd.DataFrame(
        {
            "Pregnancies": [2],
            "Glucose": [120.0],
            "BloodPressure": [80.0],
            "SkinThickness": [20.0],
            "Insulin": [85.0],
            "BMI": [28.0],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [45],
            "Outcome": [1],
        }
    )


def test_medical_feature_engineering_output(sample_patient_data):
    """Verify that clinical ratios are calculated correctly."""
    engineer = MedicalFeatureEngineer()
    transformed = engineer.transform(sample_patient_data)

    expected_homa = (120.0 * 85.0) / 405
    assert pytest.approx(transformed["HOMA_IR"].iloc[0], 0.01) == expected_homa
    assert "AgeGroup_45-60" in transformed.columns
    assert transformed["AgeGroup_45-60"].iloc[0] == 1


def test_clinical_schema_validation(sample_patient_data):
    """Verify that Pandera rejects out-of-range clinical data."""
    clinical_schema.validate(sample_patient_data)

    invalid_data = sample_patient_data.copy()
    invalid_data["Glucose"] = 900.0

    with pytest.raises(pa_errors.SchemaError):
        clinical_schema.validate(invalid_data)


def test_one_hot_encoding_consistency():
    """Ensure all age dummy columns exist for single-row inference."""
    engineer = MedicalFeatureEngineer()
    young_patient = pd.DataFrame(
        {
            "Pregnancies": [0],
            "Glucose": [100],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [50],
            "BMI": [22],
            "DiabetesPedigreeFunction": [0.3],
            "Age": [20],
        }
    )

    transformed = engineer.transform(young_patient)
    expected_cols = ["AgeGroup_30-45", "AgeGroup_45-60", "AgeGroup_60+"]
    for col in expected_cols:
        assert col in transformed.columns
        assert transformed[col].iloc[0] == 0
