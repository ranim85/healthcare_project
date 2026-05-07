"""Data loading and validation utilities."""

from pathlib import Path

import pandas as pd

from config.config import FEATURES, RAW_DATA_PATH, TARGET
from src.data.schemas import clinical_schema
from src.utils.logger import sys_logger as logger


def load_and_validate_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV data, sanitize it, and validate against the schema.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Validated and sanitized DataFrame.

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    logger.info(f"Ingesting clinical data from {path}")

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if col != TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        df = clinical_schema.validate(df)
        logger.info("Data validation successful (Pandera Schema passed)")
    except Exception as exc:
        logger.error(f"Data validation failed: {exc}")
        raise

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the dataset into features and target."""
    return df[FEATURES], df[TARGET]
