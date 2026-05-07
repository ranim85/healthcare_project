"""Centralized project configuration."""

import os
from pathlib import Path

BASE_DIR: Path = Path(__file__).parent.parent
SRC_DIR: Path = BASE_DIR / "src"
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
VISUALS_DIR: Path = BASE_DIR / "visuals"
TMP_DIR: Path = BASE_DIR / ".tmp"

for folder in [DATA_DIR, MODELS_DIR, REPORTS_DIR, VISUALS_DIR, TMP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH: Path = DATA_DIR / "healthcare_raw.csv"
TRAIN_DATA_PATH: Path = DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = DATA_DIR / "test.csv"

FEATURES: list[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ENGINEERED_FEATURES: list[str] = [
    "HOMA_IR",
    "BMI_Age_Ratio",
    "Glucose_BMI_Interaction",
    "Metabolic_Risk_Score",
    "Pregnancy_Risk",
    "Insulin_Glucose_Ratio",
    "AgeGroup_30-45",
    "AgeGroup_45-60",
    "AgeGroup_60+",
]

TARGET: str = "Outcome"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
AGE_BINS: list[int] = [0, 30, 45, 60, 120]
AGE_LABELS: list[str] = ["<30", "30-45", "45-60", "60+"]
MLFLOW_EXPERIMENT_NAME: str = "Healthcare_Production_v1"

MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "clinical_pipeline.joblib")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
SHAP_BACKGROUND_PATH: Path = MODELS_DIR / "shap_background.joblib"
LOG_PATH: Path = REPORTS_DIR / "system.log"
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")
ENABLE_MLFLOW_MODEL_LOGGING: bool = os.getenv("ENABLE_MLFLOW_MODEL_LOGGING", "0") == "1"
