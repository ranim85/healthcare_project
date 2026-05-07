"""SHAP explainability utilities."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap

from config.config import VISUALS_DIR
from src.utils.logger import sys_logger as logger


def perform_shap_analysis(
    pipeline: Any,
    X_test: pd.DataFrame,
    save_path: str = "shap_summary.png",
    max_rows: int = 250,
) -> None:
    """Generate SHAP summary plots for global interpretability."""
    logger.info("Initializing SHAP interpretability analysis")

    X_sample = X_test.sample(min(len(X_test), max_rows), random_state=42)
    X_transformed = pipeline[:-1].transform(X_sample)
    clf = pipeline.named_steps["classifier"].calibrated_classifiers_[0].estimator

    explainer = shap.Explainer(clf, X_transformed)
    shap_values = explainer(X_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / save_path)
    plt.close()

    logger.info(f"SHAP summary saved to {VISUALS_DIR / save_path}")


def get_patient_explanation(
    pipeline: Any,
    patient_df: pd.DataFrame,
    background: pd.DataFrame,
    max_background_rows: int = 100,
) -> Any:
    """Calculate SHAP waterfall values for a single patient."""
    preproc = pipeline[:-1]
    patient_transformed = preproc.transform(patient_df)

    if len(background) > max_background_rows:
        background = background.sample(max_background_rows, random_state=42)

    clf = pipeline.named_steps["classifier"].calibrated_classifiers_[0].estimator
    explainer = shap.Explainer(clf, background)
    shap_values = explainer(patient_transformed)

    return shap_values[0]
