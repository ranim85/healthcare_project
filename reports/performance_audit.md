# Performance Audit

## Scope

Benchmarks were run locally on a single-row inference request and a short training run. Results are useful for comparing future changes on the same machine. They are not capacity numbers for a hosted service.

## Inference

| Metric | Value |
| --- | ---: |
| Iterations | 200 |
| Mean latency | 104.081 ms |
| P95 latency | 108.091 ms |
| Peak Python memory | 6.006 MB |

The inference path is fast enough for local UI work and low-volume API checks. For a hosted API, latency should be measured again inside the target container and machine type.

## Training

| Metric | Value |
| --- | ---: |
| Optuna trials | 1 |
| Training time | 23.774 s |
| Peak Python memory | 48.217 MB |
| ROC-AUC | 0.7537 |
| F1 score | 0.7993 |
| Recall / sensitivity | 0.8654 |
| Precision / PPV | 0.7426 |
| Brier score | 0.1892 |

The full 10-trial run reported in the main docs produced ROC-AUC `0.7768` and Brier score `0.1804`. This file keeps the 1-trial benchmark because it is meant to measure runtime, not final model selection.

## SHAP Cost Control

SHAP summary plots sample at most 250 rows. Single-patient explanations cap background data at 100 rows. These limits keep local explanation runs bounded.

## Notes

- Benchmarks use Python-level memory tracking through `tracemalloc`.
- Container memory and CPU limits were not measured.
- Batch inference was not benchmarked.
