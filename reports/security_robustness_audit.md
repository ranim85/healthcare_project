# Security And Robustness Audit

## Scope

This review covers the checks currently implemented in the repository. It is not a security assessment for a hosted medical application.

## Implemented Controls

| Risk | Current control |
| --- | --- |
| Model path traversal | `src.models.api._safe_model_path` restricts loading to `models/`. |
| Extra API fields | Pydantic forbids unknown fields. |
| Out-of-range values | Pydantic validates clinical ranges. |
| Boolean coercion | A validator rejects booleans as measurements. |
| NaN and infinity in API | `allow_inf_nan=False` rejects non-finite payload values. |
| NaN in direct model inference | Median imputation handles missing numeric values. |
| Schema-valid extreme values | Tests cover upper-bound inputs. |
| Internal prediction failures | The API returns a generic 400 response. |
| Untrusted `joblib` artifacts | Docs and code restrict loading to trusted local artifacts. |

## Tests

- `tests/test_api_contract.py`
- `tests/test_inference_robustness.py`
- `tests/test_data_leakage.py`
- `tests/test_smoke.py`

## Remaining Gaps

- No authentication.
- No rate limiting.
- No signed model artifacts.
- No dependency vulnerability scan in CI.
- No request audit trail beyond application logs.

## Practical Next Steps

Add authentication, rate limiting, dependency scanning, signed artifacts, and request tracing before hosting this API outside a controlled environment.
