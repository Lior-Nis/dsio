# Story 7.5: Declare prediction outputs without writing validators

Status: done

## Story

As a consumer exporting a predictor,
I want DSio binary, multiclass/ordinal and regression outputs with parameterized validators and MLflow signatures,
so that I never write a normalizer or validator (cohort #24, #25, #26).

## Acceptance Criteria

1. Given each output:
   - binary threshold and probability;
   - multiclass argmax and probability, with an optional ordinal label offset;
   - regression with a declared inverse target transform and optional non-negativity;

   when its validator receives a violating prediction, then it rejects each violation class with a named error: shape, finiteness, sign, range, simplex, threshold consistency and argmax consistency.
2. Given each output inside a predictor, when exported, loaded through MLflow and used in `predict`, then outputs are identical before and after the round trip, and the Logged Model signature declares the output fields.
3. Given titanic, bike_sharing, store_sales and digit_recognizer, and the supervised fixture, when they are migrated, then their local normalizers and validators are deleted. `TensorOutput` and `validate_tensor_prediction` are removed once no consumer or fixture uses them. The self_supervised fixture keeps its embedding-norm pair as fixture code.

## Tasks / Subtasks

- [x] Task 1: `dsio.experimental.inference.outputs` (AC 1). Each output is the predictor's normalizer, and its `validator` attribute is the matching validator built from the same parameters. Consumers pass `normalizer=output, validator=output.validator`, so the two cannot disagree. Every violation raises `PredictionViolation` (a `PredictorError`), whose `kind` names the rule.
  - **`BinaryOutput(threshold=0.5)`** maps `[B]`/`[B,1]` logits to an int64 `prediction` and a sigmoid `score`. `BinaryValidator` checks shape, finiteness, range and threshold consistency.
  - **`MulticlassOutput(classes, scores=False, label_offset=0)`** produces an argmax `prediction` (plus `label_offset`) and, with `scores`, softmax probabilities as `score`.
    - With scores, the argmax is taken over them, so the two always agree; this matches Essay's ordinal output.
    - Non-finite logits are refused before the argmax, since NaN would otherwise yield a silent class.
    - `MulticlassValidator` checks shape, finiteness, range, simplex (non-negative rows summing to 1 within 1e-6) and argmax consistency.
  - **`RegressionOutput(shape, inverse=None|"expm1", non_negative=False, raw_field=None)`** returns `prediction` after the inverse transform, plus the untransformed values under `raw_field`.
    - Nothing is clamped: a negative value is a `sign` violation, not a value to hide.
    - `RegressionValidator` checks shape, dtype, finiteness and sign.
- [x] Task 2: Tests in `tests/experimental/inference/test_outputs.py`:
  - **Parity:** bit-for-bit with the deleted Titanic, Digit, Essay-style and Store normalizers.
  - **Named violations:** every violation kind, per output.
  - **MLflow round trip:** for each output, through both the `pytorch` and `pyfunc` forms. Outputs are identical, and the signature declares exactly `sample_id` plus the output fields.
  - **Validation:** configuration checks.
- [x] Task 3: Migrations (AC 3), with goldens unchanged:
  - **titanic:** `BinaryOutput`.
  - **bike_sharing:** `RegressionOutput([1], non_negative)`.
  - **store_sales:** `RegressionOutput([HORIZON], expm1, non_negative, raw_field="log_prediction")`.
  - **digit_recognizer:** `MulticlassOutput(10)`.
  - **supervised fixture:** `RegressionOutput([1])`.
  - **Common to all:** each consumer declares one `OUTPUT` config, which provenance records under `output`, replacing the separate normalizer and validator references.
  - **Legacy removed:** `TensorOutput` and `validate_tensor_prediction` are deleted, along with their frozen legacy entries and evidence.
  - **Self-supervised fixture:** keeps `EmbeddingNorm`/`validate_embedding_norm`, now with its own finiteness check.
  - **Stable tests:** the stable inference and evaluation tests use a test-local `NamePrediction` double instead of an experimental block.
- [x] Task 4: Catalog.
  - All six classes have full sections and doctested Examples.
  - Evidence records each output's real uses and each validator's uses `via` its output, plus the owner's standing approval.
  - "Prediction normalizers and validators" in the candidates register keeps FoG, ROGII, Essay and both CMI consumers.
- [x] Task 5: Review follow-ups. The adversarial review found nothing critical. Parity held for all five consumers (including Store's dropped `clamp_min(0)`, which a softplus makes a no-op, sign bits included), as did the round trip through the `pytorch` form, `pyfunc` and `predict`.
  - [x] 5.1 (important) `MulticlassOutput(scores=True)` rejected its own float16/bf16 output. The simplex check now sums in float32 with a tolerance of `max(1e-6, classes * eps)` and no relative term.
  - [x] 5.2 Wrong dtypes are their own `dtype` violation kind.
  - [x] 5.3 `PredictionViolation` pickles and deep-copies, which matters for process-based Prefect runners.
  - [x] 5.4 The standalone validators check their parameters like the outputs do.
  - [x] 5.5 `MulticlassOutput(score_field=...)` names the probability field, so CMI (`probability`) can migrate without renaming its logged signature.
  - **Recorded, not changed:** pickled predictors reference `dsio.experimental.inference.outputs.*`, so promoting these blocks to stable needs alias re-exports.

## Dev Notes

- **Parity reasoning.**
  - Titanic: `sigmoid(...).reshape(-1)` and `(score >= 0.5).to(int64)` are the same ops as before.
  - Digit: argmax over logits.
  - Bike: identity.
  - Store: `expm1` without the former `clamp_min(0)`. Its log-space outputs come from a softplus, so they are never negative, and the clamp never changed a value. The new `sign` check on `prediction` is equivalent to the old non-negativity check on `log_prediction`.
- **Why the validator lives on the output.** A predictor's validator must be an importable callable. A callable instance of an importable class qualifies and pickles into both MLflow forms. Deriving it from the output's parameters removes the threshold, class-count and shape duplication between two configs.
- **What stays local.**
  - Dense binary outputs (FoG) and scaled or per-step regression (ROGII), Story 8.6.
  - Essay's and both CMI consumers' outputs, which move with their stories.

## Dev Agent Record

### Agent Model Used

Claude Opus 5.5 (claude-opus-5-5)

### Completion Notes List

### File List

- `src/dsio/experimental/inference/outputs.py` (rewritten), `src/dsio/experimental/inference/__init__.py`
- `tests/experimental/inference/test_outputs.py` (new)
- `reference_projects/kaggle/{titanic,bike_sharing,store_sales,digit_recognizer}/{components.py,tasks/downstream.py}`
- `reference_projects/supervised/{components.py,tasks/export.py}`, `reference_projects/self_supervised/components.py`
- `tests/kaggle_portfolio/test_digit_recognizer.py`, `tests/reference_flows/test_supervised_flow.py`
- `tests/inference/{test_export.py,test_loading.py,test_predictor.py}`, `tests/eval/test_execution.py`
- `tools/catalog.py` (legacy list), `docs/component-warehouse/{catalog.md,evidence.yaml,candidates.yaml}`, `_bmad-output/implementation-artifacts/sprint-status.yaml`
