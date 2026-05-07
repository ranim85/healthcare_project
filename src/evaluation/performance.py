"""Performance benchmark utilities for training and inference."""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.config import MODEL_PATH, REPORTS_DIR
from src.models.train import run_training
from src.utils.logger import sys_logger as logger
from src.utils.sklearn import configure_sklearn_output

SAMPLE_PATIENT = pd.DataFrame(
    [
        {
            "Pregnancies": 3,
            "Glucose": 120.0,
            "BloodPressure": 80.0,
            "SkinThickness": 20.0,
            "Insulin": 100.0,
            "BMI": 30.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 40,
        }
    ]
)


def _load_or_train_pipeline() -> Any:
    """Load the local model, training one quick benchmark model if needed."""
    configure_sklearn_output()
    if not MODEL_PATH.exists():
        run_training(n_trials=1)
    return joblib.load(MODEL_PATH)


def benchmark_inference(iterations: int = 200) -> dict[str, float]:
    """Benchmark single-row inference latency and memory usage."""
    pipeline = _load_or_train_pipeline()

    tracemalloc.start()
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        pipeline.predict_proba(SAMPLE_PATIENT)
        timings.append(time.perf_counter() - start)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "iterations": float(iterations),
        "mean_latency_ms": statistics.mean(timings) * 1000,
        "p95_latency_ms": statistics.quantiles(timings, n=20)[18] * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def benchmark_training(n_trials: int = 1) -> dict[str, float]:
    """Benchmark end-to-end training time and Python memory allocation."""
    tracemalloc.start()
    start = time.perf_counter()
    metrics = run_training(n_trials=n_trials)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "training_seconds": elapsed,
        "training_peak_memory_mb": peak / (1024 * 1024),
        **metrics,
    }


def write_performance_audit(
    path: Path = REPORTS_DIR / "performance_audit.md",
    include_training: bool = False,
    n_trials: int = 1,
) -> Path:
    """Run benchmarks and write a Markdown performance audit."""
    logger.info("Starting performance benchmark audit")
    inference = benchmark_inference()
    training = benchmark_training(n_trials=n_trials) if include_training else None

    training_section = """
## Training Benchmark

Training benchmark was skipped for this report run. Execute:

```bash
python -m src.evaluation.performance --train
```
"""
    if training is not None:
        training_section = f"""
## Training Benchmark

| Metric | Value |
| --- | ---: |
| Optuna trials | {n_trials} |
| Training time | {training["training_seconds"]:.3f} s |
| Peak Python memory | {training["training_peak_memory_mb"]:.3f} MB |
| ROC-AUC | {training["ROC_AUC"]:.4f} |
| F1 Score | {training["F1_Score"]:.4f} |
| Recall / Sensitivity | {training["Recall_Sensitivity"]:.4f} |
| Precision / PPV | {training["Precision_PPV"]:.4f} |
| Brier Score | {training["Brier_Score"]:.4f} |
"""

    content = f"""# Performance Audit

## Scope

Benchmarks were run locally on a single-row inference request. Results are
useful for comparing future changes on the same machine, not for sizing a
hosted service.

```bash
python -m src.evaluation.performance --train
```

## Inference

| Metric | Value |
| --- | ---: |
| Iterations | {inference["iterations"]:.0f} |
| Mean latency | {inference["mean_latency_ms"]:.3f} ms |
| P95 latency | {inference["p95_latency_ms"]:.3f} ms |
| Peak Python memory | {inference["peak_memory_mb"]:.3f} MB |

{training_section}

## SHAP Optimization

SHAP summary analysis samples at most 250 rows, and patient explanations cap
background data at 100 rows. These limits keep local explanation runs bounded.

## Notes

- Benchmarks use Python-level memory tracking through `tracemalloc`.
- Container memory and CPU limits were not measured.
- Batch inference was not benchmarked.
"""

    path.write_text(content, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local performance benchmarks.")
    parser.add_argument("--train", action="store_true", help="Include a quick training benchmark.")
    parser.add_argument(
        "--n-trials", type=int, default=1, help="Optuna trials for training benchmark."
    )
    args = parser.parse_args()

    write_performance_audit(include_training=args.train, n_trials=args.n_trials)
