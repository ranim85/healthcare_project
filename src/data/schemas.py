"""Pandera schemas for clinical datasets."""

import pandera.pandas as pa

from config.config import TARGET

clinical_schema = pa.DataFrameSchema(
    columns={
        "Pregnancies": pa.Column(pa.Int, checks=pa.Check.greater_than_or_equal_to(0)),
        "Glucose": pa.Column(
            pa.Float,
            checks=[pa.Check.in_range(0, 500)],
            nullable=True,
        ),
        "BloodPressure": pa.Column(
            pa.Float,
            checks=[pa.Check.in_range(0, 250)],
            nullable=True,
        ),
        "SkinThickness": pa.Column(
            pa.Float,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
        ),
        "Insulin": pa.Column(
            pa.Float,
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
        ),
        "BMI": pa.Column(
            pa.Float,
            checks=[pa.Check.in_range(0, 80)],
            nullable=True,
        ),
        "DiabetesPedigreeFunction": pa.Column(
            pa.Float,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
        "Age": pa.Column(pa.Int, checks=pa.Check.in_range(0, 120)),
        TARGET: pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), nullable=True),
    },
    strict=True,
    coerce=True,
)
