"""Synthetic healthcare dataset generator."""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import RANDOM_STATE, RAW_DATA_PATH


def generate_dataset(n_samples: int = 2_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic diabetes-risk dataset.

    Args:
        n_samples: Number of patient rows to generate.
        random_state: Seed for deterministic generation.

    Returns:
        Synthetic clinical DataFrame compatible with the project schema.
    """
    rng = np.random.default_rng(random_state)

    age = rng.normal(loc=45, scale=15, size=n_samples).clip(18, 90).astype(int)
    bmi = rng.normal(loc=28, scale=6, size=n_samples).clip(15, 60).round(1)
    glucose = rng.normal(loc=120, scale=35, size=n_samples).clip(50, 300).round(1)
    blood_pressure = rng.normal(loc=80, scale=12, size=n_samples).clip(40, 140).round(1)
    insulin = rng.normal(loc=80, scale=115, size=n_samples).clip(0, 900).round(1)
    skin_thickness = rng.normal(loc=23, scale=10, size=n_samples).clip(0, 99).round(1)
    pregnancies = np.where(
        rng.random(n_samples) > 0.5,
        rng.integers(0, 15, size=n_samples),
        0,
    )
    pedigree = rng.exponential(scale=0.47, size=n_samples).clip(0.08, 2.5).round(3)

    insulin[rng.random(n_samples) < 0.35] = np.nan
    skin_thickness[rng.random(n_samples) < 0.30] = np.nan
    blood_pressure[rng.random(n_samples) < 0.05] = np.nan

    log_odds = (
        -6
        + 0.04 * glucose
        + 0.03 * bmi
        + 0.02 * age
        + 0.3 * pedigree
        + np.where(np.isnan(insulin), 0, 0.002 * insulin)
        + rng.normal(0, 0.5, n_samples)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    outcome = (rng.random(n_samples) < probability).astype(int)

    return pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
            "Outcome": outcome,
        }
    )


def save_dataset(path: Path = RAW_DATA_PATH) -> Path:
    """Generate and persist the default dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    output_path = save_dataset()
    print(f"Dataset saved to {output_path}")
