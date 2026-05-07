# Technical Audit Report

## Scope

Review date: 2026-05-07. The review covered source layout, reproducibility, data leakage, API validation, tests, performance, security checks, Docker files, and documentation.

## Status

| Area | Status | Notes |
| --- | --- | --- |
| Config | Pass | `config/config.py` owns paths, feature names, MLflow URI, and temp directory. |
| Source layout | Pass | Active code is under `src/data`, `src/features`, `src/models`, `src/evaluation`, `src/visualization`, and `src/utils`. |
| Legacy files | Pass | Old duplicate modules were removed or replaced. |
| Logging | Pass | Loguru setup is centralized. |
| Leakage control | Pass | Feature engineering learns statistics in `fit`, not from inference batches. |
| Reproducibility | Pass | Split, XGBoost, and Optuna sampler are seeded. |
| API validation | Pass | Invalid ranges, booleans, NaN/inf, and extra fields are rejected. |
| Tests | Pass | Pytest covers schema, leakage, smoke paths, corrupted API input, NaN inference, and extreme values. |
| Docker files | Partial | Files are present. Build was not validated locally because Docker is unavailable. |

## Commands Used

```bash
python -m pytest
python -m ruff check .
python -m black --check .
python -m isort --check-only .
python -m src.models.train
python -m src.evaluation.performance --train --n-trials 1
```

## Latest Test Result

```text
15 passed
```

## Latest Full Training Snapshot

```text
ROC_AUC             : 0.7768
F1_Score            : 0.8014
Recall_Sensitivity  : 0.8692
Precision_PPV       : 0.7434
Brier_Score         : 0.1804
```

## Remaining Risks

- Synthetic data limits the value of the metrics.
- `joblib` should only be loaded from trusted sources.
- MLflow is local and does not provide a promotion workflow.
- The API has no authentication, rate limiting, or request tracing.
- Subgroup metrics are not part of the standard training output.
