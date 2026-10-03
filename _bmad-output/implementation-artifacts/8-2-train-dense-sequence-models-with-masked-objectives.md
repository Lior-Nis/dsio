---
baseline_commit: ddc8e4a4aa7e5d857571acf33d75cc99f083fea7
---

# Story 8.2: Train dense sequence models with masked objectives

Status: done

## Story

As a consumer predicting per timestep,
I want a DSio objective applying any elementwise loss only where the batch `mask` is True,
so that padded and ignored positions never affect training (cohort #19).

## Acceptance Criteria

1. Given a masked objective over BCE or MSE, when masked-out prediction or target positions change value, then the loss and configured metrics are unchanged, masked positions contribute no gradient, and an all-False mask raises a clear error instead of producing NaN.
2. Given dense predictions and targets with identical shapes, when the mask is applied, then a boolean mask may match the full shape or omit only the final output axis; every other dtype, rank, extent, or broadcast would fail explicitly.
3. Given `parkinsons_fog` (BCE over `[B,T,K]`) and `rogii` (MSE over `[B,T]`, with validity represented both in `x` and the batch `mask`), when migrated, then `FogObjective` and `TvtObjective` are deleted, complete objective configurations enter provenance, logged metric names remain unchanged, and both contract tiers match the unchanged golden metrics.
4. Given warehouse admission rules, when the shared objective is added, then it is exported from `dsio.experimental.model`, both real uses are registered, the superseded candidates are removed, and catalog/admission checks pass.

## Tasks / Subtasks

- [x] Task 1: Specify the minimal masked-objective contract with failing tests (AC 1, 2).
  - [x] Prove exact parity with FoG's masked BCE on `[B,T,K]` and ROGII's masked MSE plus scaled RMSE on `[B,T]`.
  - [x] Prove masked-out prediction/target values, including non-finite values, cannot affect the loss, metrics, or gradients by selecting valid values before invoking native losses.
  - [x] Cover missing/non-tensor fields, non-boolean and all-False masks, target/prediction mismatch, invalid mask shapes, non-tensor model output, metric stages, detached metrics, importability, and canonical `ComponentConfig` resolution.
- [x] Task 2: Implement `MaskedObjective` as a native `nn.Module` (AC 1, 2).
  - [x] Configure one native loss plus optional named metrics and metric stages; reuse the Story 7.4 component construction and validation rules rather than adding another registry or result type.
  - [x] Keep the reserved fields fixed as `y` and `mask`; allow only the target dtype adaptation proven by the consumers; reject silent shape broadcasting and numeric-mask coercion.
  - [x] Leave `_sample_mean_loss` false/absent: a valid-position mean is not a sample mean, so automatic sample-normalized accumulation remains unsupported.
  - [x] Extend `RootMeanSquaredError` with a finite positive `scale=1.0` parameter so ROGII can preserve `rmse = sqrt(mse) * 20_000` without objective-specific logic.
  - [x] Keep source files focused. Because `objectives.py` is already 288 lines, either add the new block in a separate responsibility-named module or convert it to an `objectives/` package; do not grow one mixed file indefinitely. If defining modules move, update catalog/evidence references together.
- [x] Task 3: Migrate FoG and ROGII (AC 3).
  - [x] Declare one full objective `ComponentConfig` per consumer, resolve it for `DsioModule`, and record the exact mapping in training provenance.
  - [x] Delete `FogObjective` and `TvtObjective` plus unused functional-loss imports.
  - [x] Keep FoG's accumulation guard, but rename its messages to the generic masked-point reduction rather than the deleted local class.
  - [x] Make ROGII consume the batch `mask` and prove it equals the validity channel declared in `x`; do not widen the model input or change its baseline behavior.
- [x] Task 4: Govern and measure the component (AC 4).
  - [x] Add both real uses for `MaskedObjective`, add ROGII evidence for scaled `RootMeanSquaredError`, remove FoG/ROGII from `Local objectives`, remove the Story 8.2 future marker in conventions, and regenerate the catalog.
  - [x] Run admission checks and record an adversarial review before merge.
  - [x] Run `tools/consumer_metrics.py`: baseline FoG/ROGII non-ingestion lines are 801/551 (1,352 combined), and banned local definitions must fall from 28 to 26.
- [x] Task 5: Verify without changing expected behavior (AC 1-4).
  - [x] Run focused objective tests and both consumer contract suites.
  - [x] Run full pytest, Ruff check/format, mypy, import contracts, catalog/admission checks, wheel/sdist build, and distribution/consumer-flow contracts.
  - [x] Keep `tests/golden_metrics.json` byte-for-byte unchanged; no computation-change tolerance is declared for this story.

### Review Findings

- [x] [Review][Patch] Preserve axis-weight semantics when a full-shape mask flattens the output axis [src/dsio/experimental/model/masked_objective.py:86]
- [x] [Review][Patch] Enforce the objective-owned mean reduction and scalar loss contract [src/dsio/experimental/model/masked_objective.py:77]
- [x] [Review][Patch] Require configured metrics to return scalar tensors [src/dsio/experimental/model/masked_objective.py:94]
- [x] [Review][Patch] Accept the documented omitted-final-axis mask for rank-one predictions [src/dsio/experimental/model/masked_objective.py:128]
- [x] [Review][Patch] Reject an empty selected tensor even when a prefix mask contains True [src/dsio/experimental/model/masked_objective.py:86]
- [x] [Review][Patch] Validate runtime stage names instead of silently suppressing metrics [src/dsio/experimental/model/masked_objective.py:84]
- [x] [Review][Patch] Validate the reserved `x` field with an objective-specific error [src/dsio/experimental/model/masked_objective.py:84]
- [x] [Review][Patch] Cover masked non-finite predictions and robustly reject overflowing RMSE scales [tests/experimental/model/test_objectives.py:277]

## Dev Notes

- The objective contract remains `(model, batch, stage) -> {"loss": scalar, ...metrics}`. `DsioModule.log()` owns logging; the component returns native tensors only.
- `mask` is boolean with `True = valid`. It belongs to objectives and evaluation, while `x` remains the only model input. Do not add a configurable mask field, model-input mapping, task mode, base class, or result dataclass.
- Require target and prediction shapes to match exactly before masking. Accept only `mask.shape == prediction.shape` or `mask.shape == prediction.shape[:-1]`; FoG needs the latter to select timesteps while retaining its three output channels.
- Slice prediction and target before calling the configured loss or metrics. Multiplying a dense loss by a mask can let ignored NaNs poison gradients and needlessly relies on broadcasting.
- Build losses and metrics through the same validated `ComponentConfig` path as `SupervisedObjective`. The objective owns reduction parameters; stateful TorchMetrics remain unsupported.
- Metrics run on detached selected predictions and follow `metric_stages`. Preserve ROGII's `rmse` name and exact `TARGET_SCALE` conversion.
- FoG's calibration restriction to `accumulate_grad_batches=1` remains correct: equal sample counts do not imply equal valid-position counts.
- Official PyTorch BCE-with-logits and MSE losses support arbitrary equal input/target shapes and native `none`/`mean`/`sum` reductions. This story retains native mean semantics after selecting valid positions and adds no dependency.
- Do not claim that changing masked model inputs cannot influence nearby valid outputs; convolutional models can propagate context. The guarantee is that masked prediction/target positions do not contribute directly to objective values or gradients.
- Keep the component experimental despite two unrelated real uses; promotion and its compatibility decision belong to Story 11.1. The owner's standing approval covers Epic 8 experimental entry.

### Current Files and Preserved Behavior

- `src/dsio/experimental/model/objectives.py` contains Story 7.4's construction, metric, stage, dtype, and strict-shape rules. Reuse those rules and preserve every existing `SupervisedObjective` behavior.
- `reference_projects/kaggle/parkinsons_fog/components.py` currently indexes `[B,T,K]` prediction/target tensors with `[B,T]` masks and rejects all-False batches. Preserve this reduction exactly.
- `reference_projects/kaggle/parkinsons_fog/tasks/training.py` constructs the local objective twice (normal and calibration module factories) and records a string reference. Both paths must resolve the same new config.
- `reference_projects/kaggle/rogii/components.py` currently derives validity from `x[:, :, -1]`, uses masked MSE, and reports detached scaled RMSE. Story 8.1's collator emits the equivalent top-level mask; Story 8.2 makes it authoritative for the objective.
- `reference_projects/kaggle/rogii/tasks/training.py` must replace direct construction and string provenance with the full config.
- Existing model classes, collation, dataset identity, seeds, optimizer, trainer topology, downstream outputs, evaluation, and submission formatting must remain unchanged.

### Project Structure Notes

- Expected DSio changes: objective implementation under `src/dsio/experimental/model/`, its package export, and focused tests under `tests/experimental/model/`.
- Expected consumer changes: FoG/ROGII `components.py` and `tasks/training.py`, their contract tests, and the shared provenance assertion helper if needed.
- Expected governance changes: `docs/component-warehouse/{evidence.yaml,candidates.yaml,catalog.md,conventions.md}`, this story, and sprint status.
- Do not edit the historical `deletion-baseline.md`, widen `Predictor`, touch deployment/orchestration, or add a dependency.

### Previous Story Intelligence

- Story 8.1 established exact True-is-valid padding masks and shared collation for both consumers. Use those masks; do not recreate padding or validity helpers inside the objective.
- Story 8.1's review found that implicit dtype conversion and shape ambiguity must fail early. Apply the same principle to mask dtype/shape and target/prediction shapes.
- The full Story 8.1 gate passed locally and in PR #92; its unchanged golden digest is `71371a90d59bb9f51c5d2d89c2acb2e6b81cce992ba550e2e6caa0362dc530`.

### Git Intelligence Summary

- `ddc8e4a` introduced variable-length collation and is the implementation baseline.
- Story 7.4 established full objective configurations in provenance and component-driven native losses; follow that pattern rather than a consumer-specific wrapper.
- Recent stories keep one component per proven abstraction, migrate consumers in the same PR, update evidence/candidates/catalog, and run unchanged contract goldens before merge.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.2 and delivery rules]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-2 and constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — cohort #19]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — batch fields, objectives, construction]
- [Source: `_bmad-output/implementation-artifacts/7-4-train-with-dsio-objectives.md`]
- [Source: `_bmad-output/implementation-artifacts/8-1-pad-variable-length-batches-with-a-dsio-collator.md`]
- [Source: `src/dsio/experimental/model/objectives.py`]
- [Source: `reference_projects/kaggle/parkinsons_fog/components.py`]
- [Source: `reference_projects/kaggle/rogii/components.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Red phase: importing `MaskedObjective` failed before implementation; the review regression suite then failed on stage validation, full-mask axis weights, rank-one masks, and overflowing RMSE scales before their patches.
- Focused objective and FoG/ROGII contract suites passed after the review fixes.
- Final full suite: 1,233 passed, 1 deselected.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added one experimental `MaskedObjective` that selects valid prediction/target values before native mean-reduced losses and detached scalar metrics, with strict reserved-field, shape, dtype, device, stage, empty-selection, and axis-weight checks.
- Migrated FoG and ROGII to complete objective configurations, removed both local objective classes, preserved FoG accumulation safety and ROGII's scaled `rmse`, and recorded exact configs in provenance.
- Review resolved eight patch findings. Two suggestions were dismissed: generic construction cannot prove arbitrary loss semantics beyond the documented elementwise contract, and the story record was intentionally incomplete until verification finished.
- Baseline FoG/ROGII non-ingestion code was 801/551 (1,352 combined); current is 795/553 (1,348 combined). Repository banned definitions fell from 28 to 26 by deleting `FogObjective` and `TvtObjective`.
- Verification passed: focused objective tests; full FoG/ROGII suites; full `1,233 passed, 1 deselected`; Ruff; format; mypy over 116 source files; three import contracts; catalog/admission (132 tests); wheel/sdist build; and distribution/consumer-flow contracts.
- `tests/golden_metrics.json` remained unchanged at SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6b81cce992ba550e2e6caa0362dc530`.

### File List

- `_bmad-output/implementation-artifacts/8-2-train-dense-sequence-models-with-masked-objectives.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/conventions.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/parkinsons_fog/components.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/training.py`
- `reference_projects/kaggle/rogii/components.py`
- `reference_projects/kaggle/rogii/tasks/training.py`
- `src/dsio/experimental/model/__init__.py`
- `src/dsio/experimental/model/masked_objective.py`
- `src/dsio/experimental/model/objectives.py`
- `tests/experimental/model/test_objectives.py`
- `tests/kaggle_portfolio/assertions.py`
- `tests/kaggle_portfolio/test_parkinsons_fog.py`
- `tests/kaggle_portfolio/test_rogii.py`

## Change Log

- 2026-10-03: Created Story 8.2 with mask-shape, reduction, migration, provenance, and verification guardrails.
- 2026-10-03: Added and reviewed the shared masked objective, migrated both dense consumers, and admitted the component.
