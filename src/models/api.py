"""FastAPI inference entry point."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.config import FEATURES, MODEL_PATH, MODELS_DIR
from src.utils.logger import sys_logger as logger
from src.utils.sklearn import configure_sklearn_output

configure_sklearn_output()

app = FastAPI(
    title="Healthcare ML Inference API",
    description="API for diabetes risk prediction",
    version="1.1.0",
)


class PatientData(BaseModel):
    """Strict input payload for a single patient prediction."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Pregnancies: int = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=500)
    BloodPressure: float = Field(ge=0, le=250)
    SkinThickness: float = Field(ge=0, le=120)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: int = Field(ge=0, le=120)

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_values(cls, value: Any) -> Any:
        """Reject booleans, which Python otherwise treats as numeric."""
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid clinical measurements")
        return value


def _safe_model_path(path: Path = MODEL_PATH) -> Path:
    """Resolve the model path and ensure it stays inside the model directory."""
    resolved_models_dir = MODELS_DIR.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_models_dir and resolved_models_dir not in resolved_path.parents:
        raise ValueError(f"Unsafe model path outside model directory: {resolved_path}")
    return resolved_path


def load_pipeline(path: Path = MODEL_PATH) -> Any | None:
    """Load the persisted sklearn pipeline if available.

    Loading pickle/joblib artifacts is intentionally restricted to the local
    trusted model directory. Do not expose this function to user-provided paths.
    """
    try:
        pipeline = joblib.load(_safe_model_path(path))
        logger.info("Inference pipeline loaded successfully")
        return pipeline
    except Exception as exc:
        logger.error(f"Failed to load pipeline: {exc}")
        return None


pipeline = load_pipeline()


@app.get("/")
def read_root() -> dict[str, str]:
    """Return API health metadata."""
    return {"status": "healthy", "model_version": "1.1.0"}


@app.post("/predict")
def predict(data: PatientData) -> dict[str, float | int | str]:
    """Predict diabetes risk for a single patient."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded")

    try:
        input_df = pd.DataFrame([data.model_dump()])[FEATURES]
        risk_proba = float(pipeline.predict_proba(input_df)[0, 1])
        prediction = int(pipeline.predict(input_df)[0])

        return {
            "diabetes_risk_probability": round(risk_proba, 4),
            "prediction": prediction,
            "clinical_status": "High Risk" if risk_proba > 0.5 else "Low Risk",
        }
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=400, detail="Prediction failed") from exc
