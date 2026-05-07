"""Data and target drift monitoring utilities."""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from config.config import RAW_DATA_PATH, REPORTS_DIR
from src.utils.logger import sys_logger as logger


def run_drift_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_name: str = "drift_report.html",
) -> Path:
    """Generate an Evidently drift report.

    Args:
        reference_data: Baseline clinical data.
        current_data: New data to compare against the baseline.
        report_name: Output HTML report filename.

    Returns:
        Path to the saved HTML report.
    """
    logger.info("Starting drift analysis with Evidently AI")

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report_path = REPORTS_DIR / report_name
    report.save_html(str(report_path))

    logger.info(f"Drift report generated at {report_path}")
    return report_path


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    run_drift_analysis(df.iloc[:1000], df.iloc[1000:])
