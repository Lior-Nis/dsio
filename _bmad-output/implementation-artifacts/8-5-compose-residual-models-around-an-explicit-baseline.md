---
baseline_commit: c7c0e0e1a367ddc59361e25ce0f81225eae49a34
---

# Story 8.5: Compose residual models around an explicit baseline

Status: done

## Story

As a consumer with a strong baseline,
I want a zero-initialized bounded residual around a declared baseline channel of `x`,
so that a new model starts exactly at the baseline (cohort #16).

## Acceptance Criteria

1. Given time-major `[B,T,F]` input and declared baseline/validity channels, when the residual model is initialized, then its output equals the valid baseline exactly and padded output is zero.
2. Given training, when parameters update, then the learned correction remains bounded by the declared magnitude around the baseline and gradients reach the residual network.
3. Given malformed rank, feature extent, channel declarations, validity, or non-finite observed input, then the component raises a named contract error with graph-compatible asynchronous validation.
4. Given canonical `ComponentConfig`, when the residual model is resolved, then its construction is deterministic, importable, full-graph compilable, and preserves the former seeded architecture and initialization order.
5. Given a configured Predictor, when it is exported, reloaded through MLflow, and invoked on the same input, then local and loaded predictions are exactly equal.
6. Given `rogii`, when migrated, then `TvtRegressor` is deleted, the complete model configuration enters provenance, contract metrics match the unchanged golden, and RMSE does not regress the last-value baseline.
7. Given warehouse admission rules, when the block is added, then its real use is registered, the superseded local-model candidate shrinks, the catalog is regenerated, and all repository gates pass.

## Predeclared Computation Change and Tolerance

- There is no declared computation change. The residual layers, construction order, ReLU, zero-initialized final linear layer, `0.01 * tanh` bound, baseline channel, and validity multiplication remain exact.
- Contract metrics use the committed golden harness tolerance (`rel_tol=1e-6`, `abs_tol=1e-9`). `tests/golden_metrics.json` must remain byte-for-byte unchanged; no wider tolerance may be selected after execution.
- The expected golden SHA-256 is `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

## Tasks / Subtasks

- [x] Specify exact baseline initialization, correction bounds, gradients, padding safety, contract failures, deterministic config, full-graph compilation, and MLflow round-trip tests (AC 1-5).
- [x] Implement one focused residual model block; add no registry, base class, task mode, result/config dataclass, or dependency (AC 1-4).
- [x] Migrate ROGII to a complete model configuration, delete `TvtRegressor`, preserve objective/output/evaluation/submission behavior, and record exact provenance (AC 5-6).
- [x] Register evidence, shrink candidates, regenerate the catalog, run adversarial review, and measure from 553 ROGII non-ingestion lines and 20 banned definitions (AC 7).
- [x] Run full pytest, Ruff, format, mypy, import contracts, admission/catalog, build, distribution/consumer-flow contracts, and verify the golden digest (AC 1-7).

### Review Findings

- [x] [Review][Patch] Preserve the source baseline dtype so zero-initialized output is an exact identity even for higher-precision input [src/dsio/experimental/model/residual.py]
- [x] [Review][Patch] Reject bounds that overflow or underflow in the effective model dtype [src/dsio/experimental/model/residual.py]
- [x] [Review][Patch] Reject values that overflow during dtype conversion and non-finite residual logits before they can corrupt the baseline [src/dsio/experimental/model/residual.py]
- [x] [Review][Patch] Require explicit input/parameter device colocation and document the device contract [src/dsio/experimental/model/residual.py]
- [x] [Review][Patch] Validate exact validity values before any dtype conversion [src/dsio/experimental/model/residual.py]
- [x] [Review][Dismiss] Do not retain a `TvtRegressor` compatibility wrapper; the project is explicitly pre-release with no backwards-compatibility requirement.
- [x] [Review][Dismiss] TorchScript, malformed-input execution through a compiled graph, and mandatory accelerator CI are outside the declared full-graph valid-input contract.

## Dev Notes

- `x` remains the sole model input. The declared validity channel is structural and never a learned feature or objective mask.
- Preserve the former pointwise MLP exactly: `Linear(features, hidden) -> ReLU -> Linear(hidden, 1)`, with the last weight and bias zeroed before first use.
- Sanitize invalid/padded rows before the residual network and use `where` on output so arbitrary non-finite padding cannot leak through `NaN * 0`.
- Keep ROGII feature engineering and regression output/validator local. Story 8.6 owns output migration; domain feature mapping remains a candidate.
- The block remains experimental with one real use; promotion belongs to Story 11.1.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.5]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — block #16]
- [Source: `_bmad-output/implementation-artifacts/8-4-compose-token-sequence-models-with-masked-pooling.md`]
- [Source: `reference_projects/kaggle/rogii/components.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Focused verification: 102 residual, ROGII, admission, and catalog tests passed after review patches.
- Real-data run: full ROGII training, export, evaluation, and inference completed in MLflow experiment 56; training run `149b57ff3e67463f947d8379d26369b5`.
- Final verification: 1,303 passed, 1 deselected; 8 isolated distribution and consumer-flow contracts passed.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added one focused `BaselineResidual` block with an exact baseline identity, bounded learned correction, safe padding, deterministic construction, and graph-compatible value validation.
- Migrated ROGII to a complete model configuration and deleted its local `TvtRegressor` without changing its pointwise architecture, initialization order, objective, output, or submission contract.
- Added exact configured local-versus-MLflow Predictor parity and full model provenance.
- Three independent review layers found five actionable numerical/device contract issues; all were patched. Compatibility and undeclared TorchScript/compiled-invalid-input requirements were dismissed under project scope.
- ROGII non-ingestion code fell from 553 to 538 lines (-15); repository banned local definitions fell from 20 to 19 (-1).
- The full real run achieved RMSE `16.491277986392166`, beating the last-value baseline RMSE `16.533783827181423`.
- Verification passed: full pytest, Ruff, format, mypy over 120 source files, three import contracts, catalog/admission, wheel/sdist build, and distribution/consumer-flow contracts.
- `tests/golden_metrics.json` remained unchanged at SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

### File List

- `_bmad-output/implementation-artifacts/8-5-compose-residual-models-around-an-explicit-baseline.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/rogii/components.py`
- `reference_projects/kaggle/rogii/tasks/training.py`
- `src/dsio/experimental/model/__init__.py`
- `src/dsio/experimental/model/residual.py`
- `tests/experimental/model/test_residual.py`
- `tests/kaggle_portfolio/test_rogii.py`

## Change Log

- 2026-10-03: Created Story 8.5 with exact baseline initialization and no declared computation change.
- 2026-10-03: Implemented, migrated, adversarially reviewed, battle-tested on the full ROGII dataset, and verified Story 8.5.
