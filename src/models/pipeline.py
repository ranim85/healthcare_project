"""Model pipeline construction."""

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.transformers import MedicalFeatureEngineer
from src.models.factory import ModelFactory
from src.utils.sklearn import configure_sklearn_output


def create_clinical_pipeline(
    model_type: str = "xgboost",
    best_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct the end-to-end clinical pipeline.

    Args:
        model_type: Supported model identifier.
        best_params: Optional tuned parameters for the base classifier.

    Returns:
        A scikit-learn Pipeline ready for fitting.
    """
    configure_sklearn_output()
    base_clf = ModelFactory.get_model(model_type, best_params)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=5,
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("engineer", MedicalFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("classifier", calibrated_clf),
        ]
    )
