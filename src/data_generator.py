"""
data_generator.py
Generates a realistic synthetic healthcare dataset (diabetes risk indicators).
Based on the structure of the Pima Indians Diabetes Dataset (UCI / Kaggle).
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 2000

def generate_dataset(n=N):
    # --- Demographics ---
    age = np.random.normal(loc=45, scale=15, size=n).clip(18, 90).astype(int)
    bmi = np.random.normal(loc=28, scale=6, size=n).clip(15, 60).round(1)

    # --- Clinical indicators ---
    glucose = np.random.normal(loc=120, scale=35, size=n).clip(50, 300).round(1)
    blood_pressure = np.random.normal(loc=80, scale=12, size=n).clip(40, 140).round(1)
    insulin = np.random.normal(loc=80, scale=115, size=n).clip(0, 900).round(1)
    skin_thickness = np.random.normal(loc=23, scale=10, size=n).clip(0, 99).round(1)
    pregnancies = np.where(
        np.random.rand(n) > 0.5,
        np.random.randint(0, 15, size=n),
        0
    )

    # Diabetes pedigree (family history score)
    dpf = np.random.exponential(scale=0.47, size=n).clip(0.08, 2.5).round(3)

    # --- Introduce realistic missing values (MCAR / MAR pattern) ---
    # Insulin is often missing in real datasets
    missing_mask_insulin = np.random.rand(n) < 0.35
    insulin[missing_mask_insulin] = np.nan

    missing_mask_skin = np.random.rand(n) < 0.30
    skin_thickness[missing_mask_skin] = np.nan

    missing_mask_bp = np.random.rand(n) < 0.05
    blood_pressure[missing_mask_bp] = np.nan

    # --- Label: diabetes (outcome) — logistic relationship with key features ---
    log_odds = (
        -6
        + 0.04  * glucose
        + 0.03  * bmi
        + 0.02  * age
        + 0.3   * dpf
        + np.where(np.isnan(insulin), 0, 0.002 * insulin)
        + np.random.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    outcome = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Pregnancies":        pregnancies,
        "Glucose":            glucose,
        "BloodPressure":      blood_pressure,
        "SkinThickness":      skin_thickness,
        "Insulin":            insulin,
        "BMI":                bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                age,
        "Outcome":            outcome,
    })

    return df


if __name__ == "__main__":
    out = Path("data/healthcare_raw.csv")
    out.parent.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"✅  Dataset saved → {out}  |  shape: {df.shape}")
    print(df.describe().T.round(2))
