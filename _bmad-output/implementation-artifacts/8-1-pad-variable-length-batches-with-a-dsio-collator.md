---
baseline_commit: a794c93f30cdacb7e0835de142efacff6365b4ac
---

# Story 8.1: Pad variable-length batches with a DSio collator

Status: done

## Story

As a consumer with ragged sequences,
I want a DSio collator that pads declared variable-length fields, stacks fixed fields and optionally emits a True-is-valid `mask`,
so that I never write a collate function.

## Acceptance Criteria

1. Given a collator `ComponentConfig` declaring padded fields and their padding values, fixed fields, and whether to emit `mask`, when items of different lengths are collated, then padded fields reach the batch maximum, fixed fields are stacked, an emitted mask is True exactly on real positions, and `sample_id` order is preserved.
2. Given padded fields that must align within a sample, when their leading-axis lengths differ, then collation fails before padding and names the sample and fields.
3. Given the depth constraint on warehouse cohort #3, when the shared interface is reviewed, then its configuration surface is materially smaller than the three deleted consumer implementations; otherwise this story stops and records why no deep common block exists.
4. Given `essay_scoring`, `parkinsons_fog`, and `rogii`, when migrated, then their local `pad_essays`, `pad_windows`, `pad_wells`, `_arrays`, and `_targets_and_mask` helpers are deleted, evaluation arrays are built through `collate_arrays` with the same configured collator as training, and all three contract tiers match their unchanged golden metrics.
5. Given the warehouse governance rules, when the new collator is admitted, then it is exported from `dsio.experimental.data`, its full `ComponentConfig` is recorded in training provenance, all three real uses are registered, the superseded candidate is removed, and catalog/admission checks pass.

## Tasks / Subtasks

- [x] Task 1: Specify the minimal padding contract with failing tests (AC 1-3).
  - [x] Add focused tests for mixed ragged lengths, padding values, fixed-field stacking, optional True-is-valid mask emission, dtype preservation, ordered identity, empty input, missing/extra fields, and non-array or scalar padded values.
  - [x] Add an alignment failure test that names the offending `sample_id` and every mismatched padded field with its length.
  - [x] Demonstrate in this story that the configuration has only the three concepts shared by all consumers and is materially smaller than the three local functions.
- [x] Task 2: Implement the experimental DSio collator (AC 1-3).
  - [x] Add one pickle-safe configured callable under `dsio.experimental.data`; use ordinary mappings/sequences rather than a parallel config/result model.
  - [x] Pad only declared fields along axis zero, stack only declared fixed fields, preserve `sample_id`, and reject undeclared or inconsistent item shapes rather than guessing.
  - [x] Emit `mask` only when requested, with `True` for real positions and `False` for padding; reject collision with an input field named `mask`.
  - [x] Export the callable from `dsio.experimental.data` and prove it resolves from a canonical `ComponentConfig`.
- [x] Task 3: Migrate the three training paths and provenance (AC 4, 5).
  - [x] Replace `pad_essays`, `pad_windows`, and `pad_wells` with per-consumer `ComponentConfig` values resolved once for `DsioDataModule`.
  - [x] Record the complete collator config in provenance and keep models, objectives, training topology, seeds, and golden expectations unchanged.
  - [x] Delete the three local collator implementations and their now-unused imports.
- [x] Task 4: Migrate evaluation and inference arrays (AC 4).
  - [x] Replace each local `_arrays` and `_targets_and_mask` path with `collate_arrays` using the consumer's dataset and the same collator configuration as training.
  - [x] Preserve raw-target semantics, dense masks, submission ordering, and predictor input shapes without introducing a replacement helper or result dataclass.
  - [x] Update malformed-evidence tests to exercise the shared dataset/collation path rather than deleted private helpers.
- [x] Task 5: Govern, measure, and verify the migration (AC 3-5).
  - [x] Add all three real uses to `evidence.yaml`, remove `Pad collate functions` from `candidates.yaml`, regenerate the catalog, and run admission coverage.
  - [x] Run `tools/consumer_metrics.py` and record the before/current non-ingestion line count and removed local findings.
  - [x] Run focused unit/integration tests, all three consumer contract suites, the full test suite, Ruff, format, mypy, import contracts, catalog/admission checks, build, and distribution contracts without editing `tests/golden_metrics.json`.

### Review Findings

- [x] [Review][Patch] Keep FoG inference independent of target and scoring-mask columns [reference_projects/kaggle/parkinsons_fog/tasks/downstream.py:159]
- [x] [Review][Patch] Reject padding values that cannot be represented by an integer or boolean field [src/dsio/experimental/data/padding.py:142]
- [x] [Review][Patch] Reject ambiguous dynamic-axis names shared by predictor inputs and outputs [src/dsio/inference/export.py:154]
- [x] [Review][Patch] Enforce the collator's CPU guarantee for fixed fields [src/dsio/experimental/data/padding.py:147]
- [x] [Review][Patch] Exercise malformed FoG evidence through `collate_arrays` and the configured collator [tests/kaggle_portfolio/test_parkinsons_fog.py:634]
- [x] [Review][Patch] Record the mandatory depth proof and the measured consumer-code delta [8-1-pad-variable-length-batches-with-a-dsio-collator.md:28]
- [x] [Review][Patch] Complete story tasks, verification evidence, completion notes, and file inventory [8-1-pad-variable-length-batches-with-a-dsio-collator.md:25]

## Dev Notes

- Keep the public API compact: a callable configured by `padded_fields`, `fixed_fields`, and `emit_mask`. Alignment is implicit across all padded fields; do not add per-field alignment groups until a real consumer proves that need.
- Use `torch.as_tensor` plus `torch.nn.utils.rnn.pad_sequence`; do not add a dependency, registry, collator base class, result model, or universal mode switch.
- `sample_id` is governed by `collate_items`/`IdentityCollator`; the new collator must preserve it in order and remain CPU-only and pickle-safe.
- `mask` polarity is repository-wide: `True` means valid. Parkinson already supplies a domain/scoring mask and therefore pads that declared field with `False`; it does not ask the collator to emit a second mask. ROGII asks the collator to emit the padding mask. Essay keeps validity in its declared `x` channel and does not emit a top-level mask.
- Fixed fields are explicitly declared and stacked with PyTorch's native default collation. All non-identity item fields must be declared padded or fixed so configuration remains auditable and typos fail early.
- Evaluation must use Story 7.6 `collate_arrays`. Selecting the predictor input fields from its returned mapping is allowed; recreating padding loops is not.
- Keep the component experimental despite three real consumers; promotion is centralized in Story 11.1. Owner standing approval already covers Epic 8 experimental entries.
- If the implementation file becomes large, promote it to a same-named package rather than accumulating unrelated classes in one module.

### Consumer Configuration

- Essay: padded `x=0`, fixed `y`, no emitted mask. Use an inputs-only `StoredItems` config for export/evaluation/inference because test samples have no target.
- Parkinson FoG: padded `x=0.0`, `y=0.0`, `mask=False`; no emitted mask. Keep the current local dataset until Story 8.3 moves normalization into DSio.
- ROGII: padded `x=0.0`, `y=0.0`; emit `mask`. Scale collated normalized `y` for evaluation exactly as before.

### Project Structure Notes

- Expected DSio files: `src/dsio/experimental/data/padding.py`, `src/dsio/experimental/data/__init__.py`, and focused tests under `tests/experimental/data/`.
- Expected consumer files: the three `components.py`, `tasks/training.py`, and `tasks/downstream.py` modules plus their contract tests.
- Expected governance files: `docs/component-warehouse/evidence.yaml`, `docs/component-warehouse/candidates.yaml`, generated `catalog.md`, this story, and sprint status.
- Do not change the stable `dsio.data.loading` surface or the `dsio.eval` import contract.

### Previous Story Intelligence

- Story 7.6 introduced `collate_arrays`, which already runs the dataset and custom collator behind the same identity/cardinality guard as `DsioDataModule`.
- `collate_arrays` batches the requested roster at once and returns NumPy arrays; it must receive the resolved padding collator for sequence consumers.
- Story 7.7 closed Epic 7 with exact parity and hardened evidence recording. Set Epic 7 to done when Story 8.1 begins.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.1]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — FR4, FR9, CAP-8, CAP-9]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — cohort #3]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md`]
- [Source: `_bmad-output/implementation-artifacts/7-6-build-evaluation-arrays-from-the-training-collation.md`]
- [Source: `src/dsio/data/loading/collation.py`]
- [Source: `src/dsio/experimental/data/arrays.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Implementation Plan

- Add the smallest configured pad/stack callable test-first.
- Replace the three local collators and downstream padding loops with the shared data path.
- Update evidence and run every contract and repository gate before review.

### Debug Log References

- Red phase: the new padding suite initially failed at import; dynamic-axis tests initially failed because `log_predictor` accepted no declaration; each review regression failed before its patch.
- Contract-tier sequence suites passed with unchanged golden metrics before and after review.
- Final full suite: 1,225 passed, 1 deselected.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added `PadCollator` with exactly three configuration concepts: padded fields/values, fixed fields, and mask emission. Those three declarations replace 46 lines across the three local collators while centralizing identity, CPU, alignment, dtype, padding, stacking, and mask invariants without task modes or a parallel config model.
- Deleted all three local collators and all five downstream `_arrays`/`_targets_and_mask` helpers; training and downstream provenance now record the full dataset/collator configuration.
- Added explicit variable axes to native MLflow signatures so batch-maximum padding remains valid across export, evaluation, and inference. FoG uses a target-free input path for export/inference.
- Story-baseline non-ingestion code for Essay/FoG/ROGII was 1,840 lines; current is 1,891 (`+51`). The increase is explicit component/provenance/dynamic-shape wiring that Epic 10 is planned to centralize. The material depth result is the three-concept call surface and removal of the three banned collator definitions; repository banned definitions fell from 31 to 28.
- Review resolved seven patch findings. Two suggestions were dismissed: fixed-width FoG evaluation contradicts batch-maximum AC1, and task-specific trailing-shape options would expand the generic API beyond the proven shared contract.
- Verification passed: focused sequence/core suites; full `1,225 passed, 1 deselected`; Ruff; format; mypy over 115 source files; three import contracts; catalog/admission (112 tests); wheel/sdist build; and nine distribution/import tests.
- `tests/golden_metrics.json` remained unchanged at SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6b81cce992ba550e2e6caa0362dc530`.

### File List

- `_bmad-output/implementation-artifacts/8-1-pad-variable-length-batches-with-a-dsio-collator.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/essay_scoring/components.py`
- `reference_projects/kaggle/essay_scoring/tasks/downstream.py`
- `reference_projects/kaggle/essay_scoring/tasks/training.py`
- `reference_projects/kaggle/parkinsons_fog/components.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/downstream.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/training.py`
- `reference_projects/kaggle/rogii/components.py`
- `reference_projects/kaggle/rogii/tasks/downstream.py`
- `reference_projects/kaggle/rogii/tasks/training.py`
- `src/dsio/experimental/data/__init__.py`
- `src/dsio/experimental/data/padding.py`
- `src/dsio/inference/export.py`
- `tests/experimental/data/test_padding.py`
- `tests/inference/test_export.py`
- `tests/kaggle_portfolio/assertions.py`
- `tests/kaggle_portfolio/test_essay_scoring.py`
- `tests/kaggle_portfolio/test_parkinsons_fog.py`
- `tests/kaggle_portfolio/test_rogii.py`

## Change Log

- 2026-10-03: Created Story 8.1 with minimal collator and migration constraints.
- 2026-10-03: Added the shared collator, migrated all three sequence consumers, and admitted the component.
- 2026-10-03: Resolved all adversarial-review findings and completed repository-wide verification.
