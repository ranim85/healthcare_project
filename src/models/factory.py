"""Clinical model factory and optimization."""

from typing import Any

import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config.config import RANDOM_STATE
from src.utils.logger import sys_logger as logger


class ModelFactory:
    """Factory for clinical models with integrated hyperparameter optimization."""

    @staticmethod
    def get_model(
        model_type: str = "xgboost",
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Return an initialized model.

        Args:
            model_type: Model identifier.
            params: Optional estimator parameters.

        Returns:
            A configured sklearn-compatible classifier.

        Raises:
            ValueError: If the model type is unsupported.
        """
        if params is None:
            params = {"random_state": RANDOM_STATE}

        if model_type == "xgboost":
            return xgb.XGBClassifier(eval_metric="logloss", n_jobs=1, **params)
        if model_type == "random_forest":
            return RandomForestClassifier(**params)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def optimize_xgboost(X: Any, y: Any, n_trials: int = 20) -> dict[str, Any]:
        """Run Optuna optimization for XGBoost.

        Args:
            X: Training features transformed inside the training split.
            y: Training labels.
            n_trials: Number of Optuna trials.

        Returns:
            Best hyperparameters found by Optuna.
        """

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": RANDOM_STATE,
            }

            clf = ModelFactory.get_model("xgboost", params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            score = cross_val_score(clf, X, y, cv=skf, scoring="f1").mean()
            return float(score)

        logger.info(f"Starting HPO for XGBoost ({n_trials} trials)")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Optimization complete. Best F1: {study.best_value:.4f}")
        return dict(study.best_params)
