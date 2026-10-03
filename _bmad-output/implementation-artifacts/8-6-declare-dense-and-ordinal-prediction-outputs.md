---
baseline_commit: a1ae0087ee25e99f419a4ea03588e7011306a7b4
---

# Story 8.6: Declare dense and ordinal prediction outputs

Status: done

## Story

As a consumer exporting sequence predictions,
I want DSio outputs to cover dense per-timestep binary and regression shapes and ordinal label offsets,
so that no sequence consumer writes a normalizer or validator (cohort #24, #25, #26).

## Acceptance Criteria

1. Given dense binary logits `[B,T,K]`, when configured with the per-sample shape `[None,K]`, then the output returns same-shape int64 predictions and probabilities using the declared threshold and field name.
2. Given dense regression values `[B,T]`, when configured with per-sample shape `[None]` and a positive finite scale, then the output returns same-shape scaled predictions.
3. Given ordinal logits `[B,K]` and a label offset, then the existing multiclass output returns labels in the offset range and probability scores with exact argmax consistency.
4. Given malformed shape, dtype, finiteness, sign, range, threshold, simplex, or argmax relationships, then the matching validator raises `PredictionViolation` naming the violation class.
5. Given every output in a configured Predictor, when exported and loaded through MLflow with dynamic time axes, then local, native, and pyfunc outputs and declared output fields are identical.
6. Given `essay_scoring`, `parkinsons_fog`, and `rogii`, when migrated, then their local output classes and validators are deleted, one complete `OUTPUT` configuration enters provenance, golden metrics and submission bytes remain unchanged, and configured local-versus-MLflow output parity holds.
7. Given warehouse governance, when the migration completes, then the three output/validator real uses are registered, the candidate register shrinks to CMI only, the generated catalog is current, and all repository gates pass.

## Predeclared Computation Change and Tolerance

- There is no declared computation change: Essay keeps `softmax -> argmax + 1`; FoG keeps `sigmoid -> >= 0.5` with the `probability` field; ROGII keeps multiplication by `20_000`.
- Contract metrics use the committed golden harness tolerance (`rel_tol=1e-6`, `abs_tol=1e-9`). `tests/golden_metrics.json` must remain byte-for-byte unchanged.
- The expected golden SHA-256 is `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.
- Local-versus-exported predictions and submission bytes require exact equality.

## Tasks / Subtasks

- [x] Specify dense binary/regression shape contracts, ordinal offsets, named failures, dynamic-axis export, and exact round-trip tests (AC 1-5).
- [x] Turn the oversized inference outputs module into a focused package and extend only the existing output blocks; add no parallel result model, registry, base class, mode enum, or dependency (AC 1-4).
- [x] Migrate Essay, FoG, and ROGII to complete `OUTPUT` configurations; delete six local normalizer/validator definitions and preserve exact output fields and bytes (AC 5-6).
- [x] Register real uses, shrink candidates, regenerate the catalog, run three-layer adversarial review, and measure from 1,826 combined non-ingestion lines and 19 banned definitions (AC 7).
- [x] Run full pytest, Ruff, format, mypy, import contracts, catalog/admission, build, distribution/consumer-flow contracts, and verify the golden digest (AC 1-7).

## Dev Notes

- Use one per-sample shape vocabulary across binary and regression outputs: positive integers are fixed extents and `None` is a dynamic extent. Batch remains implicit.
- Preserve BinaryOutput's scalar `[B]`/`[B,1]` behavior when no shape is declared.
- Scaling belongs in RegressionOutput because it converts target space to prediction space; validation remains on the final prediction.
- Keep each output's validator derived from the same configured output instance (`output.validator`).
- Keep output components experimental; promotion belongs to Story 11.1.
- Story 8.7 records representative-scale parity. Story 8.6 may produce those real runs but does not duplicate a second parity abstraction.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.6]
- [Source: `_bmad-output/implementation-artifacts/7-5-declare-prediction-outputs-without-writing-validators.md`]
- [Source: `_bmad-output/implementation-artifacts/8-5-compose-residual-models-around-an-explicit-baseline.md`]
- [Source: `src/dsio/experimental/inference/outputs.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- `uv run pytest --ignore=tests/test_distribution.py --ignore=tests/test_built_distribution.py --ignore=tests/test_project_flow.py` — 1,314 passed, 1 deselected.
- `uv run pytest -q tests/test_distribution.py tests/test_built_distribution.py tests/test_project_flow.py` — 8 passed against the built distribution.
- `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, `uv run lint-imports`, and `uv run python tools/catalog.py --check` — all green.
- `uv build` — source distribution and wheel built successfully.
- Full-data MLflow runs: Essay inference `e8277e41ca3a4957b1b74c968234383b`, FoG inference `a6752316da6244e9a5790b188b5c5391`, ROGII inference `4f28d9a464dc4900b4a9689cbde6707b`.
- Golden metrics SHA-256 remained `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

### Completion Notes List

- Split the 456-line output module into focused binary, multiclass, regression, and shared-contract modules while preserving the public inference imports.
- Added one per-sample shape vocabulary with positive fixed extents and dynamic `None` extents; dense binary and regression exports were exercised at a second sequence length through local, native MLflow, and pyfunc prediction paths.
- Hardened validation for raw and normalized finiteness, floating dtypes, non-empty dynamic extents, probability/simplex bounds, threshold/argmax/transform consistency, paired tensor shapes, scale bounds, and int64-safe ordinal offsets.
- Replaced the six Essay, FoG, and ROGII local normalizer/validator definitions with configured DSio blocks and recorded the complete output configuration in downstream provenance.
- Reduced the three consumers from 1,826 to 1,761 non-ingestion lines and reduced banned local definitions from 19 to 13. The remaining prediction candidate is CMI only.
- Full-data metrics remained stable: Essay accuracy `0.44929797191887677`, QWK `0.47384293867867033`; FoG mean average precision `0.22076385159247158`; ROGII RMSE `16.491277986392166`.
- Essay and ROGII full-data submission payload digests remained exactly `8e54f17a...20588` and `81428f91...5f94`. An independently retrained FoG model preserved ranking metrics but produced mean absolute probability drift `6.61e-6` (p99 `3.80e-5`); exact same-logit and same-checkpoint output parity is covered by the contract and MLflow round-trip tests.

### Review Findings

- [x] Tighten low-precision multiclass simplex validation so tolerance does not grow with class count.
- [x] Reject non-finite and non-floating raw logits/values before sigmoid, softmax, or inverse transforms can hide them.
- [x] Validate regression raw-to-prediction transform consistency and full paired tensor shapes.
- [x] Reject zero-length dynamic extents, int64-overflowing ordinal ranges, and non-finite, non-positive, or unrepresentable regression scales.
- [x] Exercise binary and regression MLflow round trips at a different dynamic sequence length.
- [x] Add non-default threshold/scale boundaries, regenerate the catalog, and register real inference evidence.

### File List

- `_bmad-output/implementation-artifacts/8-6-declare-dense-and-ordinal-prediction-outputs.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `src/dsio/experimental/inference/outputs/{__init__,contracts,binary,multiclass,regression}.py` (replaces `outputs.py`)
- `reference_projects/kaggle/{bike_sharing,digit_recognizer,essay_scoring,parkinsons_fog,rogii,store_sales}/components.py`
- `reference_projects/kaggle/{essay_scoring,parkinsons_fog,rogii}/tasks/downstream.py`
- `reference_projects/supervised/components.py`
- `tests/experimental/inference/test_outputs.py`
- `tests/kaggle_portfolio/{assertions,test_essay_scoring,test_parkinsons_fog,test_rogii}.py`
- `tests/reference_flows/test_supervised_flow.py`

## Change Log

- 2026-10-03: Created Story 8.6 with exact output parity and one shared dynamic-shape vocabulary.
- 2026-10-03: Implemented, adversarially reviewed, battle-tested on three full Kaggle datasets, and completed Story 8.6.
