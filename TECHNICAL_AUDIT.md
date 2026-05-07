# Technical Audit

This review records the current engineering state of the repository. It is intentionally blunt: the code is in good shape for a personal ML engineering project, but it is not clinical software.

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Tests | Pass | `15 passed` locally |
| Lint/format | Pass | Ruff, Black, and isort pass |
| Training | Pass | `python -m src.models.train` writes the model artifact |
| Reproducibility | Mostly pass | Split, XGBoost, and Optuna are seeded |
| API validation | Pass | Pydantic rejects malformed and out-of-range payloads |
| Missing values | Pass | The sklearn imputer handles NaN values at inference |
| Extreme values | Pass | Tests cover schema-valid upper-bound values |
| Model path handling | Pass | API restricts model loading to `models/` |
| Docker files | Present | Docker was not available locally for build validation |

## Latest Training Run

```text
ROC_AUC             : 0.7768
F1_Score            : 0.8014
Recall_Sensitivity  : 0.8692
Precision_PPV       : 0.7434
Brier_Score         : 0.1804
```

These numbers are from the synthetic local dataset. They are useful for regression checks, not for clinical claims.

## What Is Working Well

- Package boundaries are clear enough to navigate quickly.
- Training and serving use the same sklearn pipeline.
- Leakage around engineered features is explicitly tested.
- API input validation covers common bad payloads.
- Local development commands are short and repeatable.
- The docs state the medical limitations instead of burying them.

## Remaining Debt

- There is no external validation dataset.
- No subgroup metrics are generated as part of the standard training run.
- No confidence intervals are reported for model metrics.
- The API response is not yet a typed Pydantic response model.
- MLflow is local only; there is no promotion or rollback process.
- Docker config has not been built in this environment because Docker is not installed.
- No dashboard screenshot is committed yet.

## Security Notes

`joblib` artifacts are only safe when they come from a trusted source. The API prevents user-provided model paths, but a real deployment should use signed artifacts or a safer persistence format.

There is no authentication, rate limiting, or request tracing. That is acceptable for local inspection, but not for a hosted service with real users.

## Recommended Next Work

1. Add subgroup metrics and calibration plots to the training output.
2. Add a typed API response model.
3. Add a GitHub Actions artifact upload for reports.
4. Test Docker build in CI.
5. Add a real Streamlit screenshot under `docs/assets/`.
