"""Model training entry point."""

import tempfile

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from config.config import (
    ENABLE_MLFLOW_MODEL_LOGGING,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    SHAP_BACKGROUND_PATH,
    TEST_SIZE,
    TMP_DIR,
)
from src.data.loader import load_and_validate_data, split_features_target
from src.evaluation.metrics import calculate_clinical_metrics, print_evaluation_report
from src.models.factory import ModelFactory
from src.models.pipeline import create_clinical_pipeline
from src.utils.logger import sys_logger as logger
from src.utils.sklearn import configure_sklearn_output


def run_training(n_trials: int = 10) -> dict[str, float]:
    """Execute the full training lifecycle.

    Args:
        n_trials: Number of Optuna trials for XGBoost tuning.
    """
    tempfile.tempdir = str(TMP_DIR)
    configure_sklearn_output()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Clinical_System_Refactor"):
        logger.info("Initializing training pipeline")

        df = load_and_validate_data()
        X, y = split_features_target(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        pre_pipeline = create_clinical_pipeline()[:-1]
        X_train_transformed = pre_pipeline.fit_transform(X_train)
        best_params = ModelFactory.optimize_xgboost(
            X_train_transformed,
            y_train,
            n_trials=n_trials,
        )
        mlflow.log_params(best_params)

        logger.info("Fitting final end-to-end pipeline")
        final_pipeline = create_clinical_pipeline(best_params=best_params)
        final_pipeline.fit(X_train, y_train)

        y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_clinical_metrics(y_test, y_prob)
        print_evaluation_report(metrics)
        mlflow.log_metrics(metrics)

        joblib.dump(final_pipeline, MODEL_PATH)
        shap_bg = X_train_transformed.sample(100, random_state=RANDOM_STATE)
        joblib.dump(shap_bg, SHAP_BACKGROUND_PATH)

        if ENABLE_MLFLOW_MODEL_LOGGING:
            try:
                mlflow.sklearn.log_model(final_pipeline, "model")
            except PermissionError as exc:
                logger.warning(f"MLflow model artifact logging skipped: {exc}")

        logger.info(f"System deployment ready. Artifacts at {MODEL_PATH}")
        return metrics


if __name__ == "__main__":
    run_training()
