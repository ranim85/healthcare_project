"""Clinical feature engineering transformers."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import AGE_BINS, AGE_LABELS


class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for clinical feature engineering."""

    metabolic_columns: tuple[str, ...] = ("Glucose", "BMI", "Insulin", "BloodPressure")

    def __init__(
        self,
        age_bins: list[int] | None = None,
        age_labels: list[str] | None = None,
    ) -> None:
        """Initialize the transformer.

        Args:
            age_bins: Numeric bin edges used for age grouping.
            age_labels: Labels corresponding to the configured bins.
        """
        self.age_bins = AGE_BINS if age_bins is None else age_bins
        self.age_labels = AGE_LABELS if age_labels is None else age_labels

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MedicalFeatureEngineer":
        """Learn train-only statistics used by the metabolic risk score.

        Args:
            X: Training feature matrix after imputation.
            y: Optional target vector, unused.

        Returns:
            The fitted transformer.
        """
        numeric = self._to_dataframe(X)
        self.metabolic_min_ = numeric.loc[:, self.metabolic_columns].min()
        self.metabolic_range_ = (
            numeric.loc[:, self.metabolic_columns].max() - self.metabolic_min_
        ).replace(0, 1e-6)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply clinical feature engineering.

        Args:
            X: Input patient data.

        Returns:
            DataFrame with original and engineered features.
        """
        transformed = self._to_dataframe(X).copy()

        if not hasattr(self, "metabolic_min_"):
            self.fit(transformed)

        transformed["HOMA_IR"] = (transformed["Glucose"] * transformed["Insulin"]) / 405
        transformed["BMI_Age_Ratio"] = transformed["BMI"] / (transformed["Age"] + 1)
        transformed["Glucose_BMI_Interaction"] = transformed["Glucose"] * transformed["BMI"]
        transformed["Insulin_Glucose_Ratio"] = transformed["Insulin"] / (transformed["Glucose"] + 1)

        risk_components = (transformed.loc[:, self.metabolic_columns] - self.metabolic_min_) / (
            self.metabolic_range_ + 1e-6
        )
        transformed["Metabolic_Risk_Score"] = risk_components.mean(axis=1)
        transformed["Pregnancy_Risk"] = transformed["Pregnancies"] * (transformed["Age"] / 40)

        age_groups = pd.cut(
            transformed["Age"],
            bins=self.age_bins,
            labels=self.age_labels,
            right=False,
            include_lowest=True,
        )
        age_dummies = pd.get_dummies(age_groups, prefix="AgeGroup", drop_first=True)

        expected_cols = [f"AgeGroup_{label}" for label in self.age_labels[1:]]
        for col in expected_cols:
            if col not in age_dummies.columns:
                age_dummies[col] = 0

        age_dummies = age_dummies[expected_cols].astype(int)
        return pd.concat([transformed, age_dummies], axis=1)

    def _to_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X as a DataFrame and reject lossy ndarray inputs."""
        if isinstance(X, pd.DataFrame):
            return X

        raise TypeError(
            "MedicalFeatureEngineer requires pandas DataFrame input. "
            "Call configure_sklearn_output() before building sklearn pipelines."
        )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer to drop non-modeled columns while keeping them during engineering."""

    def __init__(self, columns: list[str]) -> None:
        """Initialize the dropper.

        Args:
            columns: Columns to remove during transform.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        """No-op fit method for sklearn compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop configured columns if present."""
        return X.drop(columns=self.columns, errors="ignore")
