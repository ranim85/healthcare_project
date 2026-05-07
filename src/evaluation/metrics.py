"""Clinical model metric utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def calculate_clinical_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate metrics relevant for medical decision support."""
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "F1_Score": float(f1_score(y_true, y_pred)),
        "Recall_Sensitivity": float(recall_score(y_true, y_pred)),
        "Precision_PPV": float(precision_score(y_true, y_pred)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
    }


def print_evaluation_report(
    metrics: dict[str, float],
    title: str = "Clinical Performance",
) -> None:
    """Print a formatted metrics report."""
    print(f"\n{'=' * 10} {title} {'=' * 10}")
    for key, value in metrics.items():
        print(f"{key:20}: {value:.4f}")
    print("=" * 40)
