# Story 7.2: Fit standardization on training-role samples

Status: review

## Story

As a consumer normalizing inputs,
I want DSio to fit standardization statistics on the training role and inject them into a standardization slot, with the fitted values logged as evidence,
so that I cannot leak validation data into normalization (cohort #4a, #9a).

## Acceptance Criteria

1. Given a split manifest, fold and store, when the statistic fitter runs, then it computes per-feature mean and standard deviation over training-role samples only (NaN-aware), and logs the values as an evidence artifact referenced from provenance.
2. Given a validation-role sample whose values change, when the fitter reruns, then the fitted statistics are identical (leakage test).
3. Given bike_sharing and digit_recognizer, when they are migrated, then bike_sharing's `_scaler` and model buffers, and digit_recognizer's `ScalePixels`, are replaced by the fixed standardization slot (fitted or constant), and contract tiers match the golden metrics.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.data.fitting:fit_standardization(store, manifest, *, fold, role="train", observed=False)` (AC 1).
  - It takes the role's samples **from the manifest**, so a caller cannot hand it held-out identities.
  - It computes float64 mean and population std over every row, mapping zero-variance features to scale 1. `observed=True` ignores NaN per feature and refuses features with no observed value.
  - It returns the statistics plus the exact identities used.
- [x] Task 2: `record_fitted(run_id, name, fitted)` logs `fitted/<name>.json` and returns that **run-relative** path (AC 1). Because a run-specific URI would make every replay's execution identity differ, provenance records the path plus the values.
- [x] Task 3: `dsio.experimental.model.standardization:Standardize(mean, scale, axis=-1)` computes exact `(x - mean) / scale` along a declared feature axis, casting integer inputs to float32 (AC 3).
- [x] Task 4: Tests (AC 2), in `tests/experimental/data/test_fitting.py`:
  - **Leakage:** setting every held-out sample to 1000 leaves the fitted mean identical.
  - **NaN handling:** `observed` ignores NaN, and an all-NaN feature raises.
  - **Bit parity:** `Standardize` matches Bike Sharing's former buffer arithmetic and Digit's former `x.float() / 255.0` exactly.
  - **Validation:** axis, feature count, lengths and zero scale are checked.
  - **Evidence:** the artifact path is relative.
- [x] Task 5: Migrations (AC 3), with goldens unchanged:
  - **bike_sharing:** `_scaler` is deleted. `fit_standardization` and `record_fitted` feed `DemandRegressor`, which now delegates to `Standardize`; the rest of the model moves to the warehouse composition in Story 7.3. Provenance records `standardization` (artifact path, role, fold, sample IDs, values).
  - **digit_recognizer:** `ScalePixels` is deleted. The predictor preprocessor is `Standardize(mean=[0], scale=[255])`.
- [x] Task 6: Catalog. All three blocks have full sections and doctested Examples. Their evidence lists real uses (`Standardize`: Bike Sharing and Digit Recognizer; the fitter and recorder: Bike Sharing) and the owner's standing approval. The candidates register now lists only CMI (both consumers) for statistics and weights (Stories 9.2 and 9.3).
- [x] Task 7: Review follow-ups. The adversarial review found nothing critical. It verified that the fitter reads only the manifest's `train` assignments, so purged and discarded samples never enter, and that `Standardize` is bit-identical to the former arithmetic.
  - [x] 7.1 Bike provenance records the whole fitted mapping, including `observed`, next to the artifact path. Two fits differing only in `observed` no longer share an identity.
  - [x] 7.2 The stale `scaler_fit_ids` result and export key is renamed `standardization_sample_ids`.
  - [x] 7.3 The Bike contract test now checks the `fitted/standardization.json` evidence that `record_fitted`'s catalog entry cites.
  - [x] 7.4 The leakage test asserts the tampered store's manifest assigns the same samples and compares `scale` as well as `mean`.

## Dev Notes

- **Parity reasoning.** The fitter repeats `_scaler`'s exact operations: concatenate rows, cast to float64, `mean`/`std`, map 0 to 1. `Standardize` builds the same float32 buffers and broadcast shape, and draws no randomness, so `DemandRegressor`'s Linear initialization is unchanged. For 8-bit pixels, `x - 0.0` is exact and the division is the same IEEE operation.
- **What stays local.** The CMI consumers' observed-mask statistics, which read mask columns rather than NaN, are Story 9.2.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

- Gates: the bike and digit contract tests (14) and the fitting and items unit tests (21) pass, as does the catalog (22 tests).

### File List

- `src/dsio/experimental/data/fitting.py`, `src/dsio/experimental/model/standardization.py` (new); experimental `data` and `model` `__init__.py`
- `tests/experimental/data/test_fitting.py` (new)
- `reference_projects/kaggle/bike_sharing/{components.py,tasks/training.py}`
- `reference_projects/kaggle/digit_recognizer/{components.py,tasks/training.py}`
- `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
