import pandas as pd
import pytest
from sklearn import get_config

from src.features.transformers import MedicalFeatureEngineer
from src.models.pipeline import create_clinical_pipeline


def test_metabolic_score_is_independent_from_inference_batch():
    """The transformer must not learn min/max values from inference rows."""
    train = pd.DataFrame(
        {
            "Pregnancies": [0, 4],
            "Glucose": [80.0, 180.0],
            "BloodPressure": [60.0, 100.0],
            "SkinThickness": [18.0, 35.0],
            "Insulin": [40.0, 220.0],
            "BMI": [20.0, 42.0],
            "DiabetesPedigreeFunction": [0.2, 1.0],
            "Age": [25, 55],
        }
    )
    patient = train.iloc[[0]].copy()
    extreme_patient = train.iloc[[1]].copy()

    engineer = MedicalFeatureEngineer().fit(train)
    alone_score = engineer.transform(patient)["Metabolic_Risk_Score"].iloc[0]
    batched_score = engineer.transform(pd.concat([patient, extreme_patient], ignore_index=True))[
        "Metabolic_Risk_Score"
    ].iloc[0]

    assert alone_score == pytest.approx(batched_score)


def test_pipeline_configures_pandas_output():
    """Pipeline creation should fix sklearn pandas output for every entry point."""
    create_clinical_pipeline()
    assert get_config()["transform_output"] == "pandas"
