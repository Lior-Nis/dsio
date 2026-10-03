---
baseline_commit: 871e31b0ccbc6cb5100a0c4e62f106c2e685d30e
---

# Story 8.3: Compose dense signal models with layout adapters and per-instance standardization

Status: done

## Story

As a consumer of multichannel signals,
I want a dense per-timestep Conv1d encoder, explicit signal-layout adapters, and per-instance standardization as composable blocks,
so that the signal model is configuration only (cohort #9b, #11, #13).

## Acceptance Criteria

1. Given time-major `[B,T,C]` input, when configured layout, standardization, dense convolution, and inverse layout stages run, then the output is `[B,T,K]`, temporal length is preserved, and rank/channel/time failures name the mismatched axis.
2. Given padded ragged input with a declared validity channel inside `x`, when per-instance statistics are computed, then padding is excluded, zero-variance channels remain finite, padding becomes normalized-space zero, and changing padded values cannot change real outputs.
3. Given deterministic standardization in the configured model preprocessing path, when a Predictor is exported, reloaded through MLflow, and invoked, then training and loaded inference use the same component definition and produce identical predictions.
4. Given `parkinsons_fog`, when migrated, then `FogDetector` and `normalize_signal` are deleted, its model/dataset/preprocessing are full `ComponentConfig` mappings, and the contract tier remains within the predeclared tolerance without changing the golden file.
5. Given warehouse admission rules, when the blocks are added, then every public callable is exported, its real use is registered, superseded candidate entries shrink, and catalog/admission checks pass.

## Predeclared Computation Change and Tolerance

- Normalization moves from NumPy before collation to a Torch model-preprocessing stage after collation. A declared True-on-real validity channel in `x` makes the result independent of batch padding; the dense block consumes it without presenting it to learned convolutions.
- The shared standardizer uses population variance (`correction=0`) and `scale.clamp_min(1e-6)`, matching the former NumPy definition. Real-position normalized tensors may differ only by floating reduction order: tests allow `rtol=1e-6`, `atol=1e-6`.
- Contract metrics use the already committed golden harness tolerance (`rel_tol=1e-6`, `abs_tol=1e-9`). `tests/golden_metrics.json` must remain byte-for-byte unchanged. No wider tolerance may be selected after observing results.
- Conv1d modules are constructed in the former order, with no parameter-bearing stage before them, so seeded parameter initialization is expected to remain exact.

### Representative Evidence Correction

- Story 8.7 found that a generic Torch sum did not preserve the declared NumPy float32 reduction tolerance on realistic 512-timestep windows. The clean zero-tolerance failure is preserved in MLflow.
- `InstanceStandardize` now uses a batched native Torch matrix-vector reduction, which preserves GPU execution while matching the former CPU NumPy accumulation within the declared tolerance for the FoG input distribution. An explicit `reduction="numpy"` mode reproduces audited NumPy float32 migrations exactly; it is not the default because it copies through CPU and is non-differentiable.
- Metric isolation showed that reduction order was not the final FoG parity cause. The legacy convolution allowed hidden activations in structural padding to influence real boundary positions, while `DenseConv1d` safely isolates padding by default. Its explicit `isolate_padding=False` migration mode preserves that legacy arithmetic; the safe default is unchanged.

## Tasks / Subtasks

- [x] Task 1: Specify the signal-block contracts with failing tests (AC 1-3).
  - [x] Prove non-square time-major/channel-first round trips, contiguity, variable time extents, and named rank/channel/optional-time extent errors.
  - [x] Prove the dense Conv1d block exactly matches the former two-convolution architecture under one seed and preserves `[B,C,T] -> [B,K,T]`.
  - [x] Prove population-standardization parity, zero variance, dtype/device behavior, input immutability, deterministic output, validity-channel preservation, padding invariance, invalid masks, and no-observed-value errors.
  - [x] Prove canonical `ComponentConfig` resolution, serialization/importability, docstring admission, and a Predictor export/load/predict round trip.
- [x] Task 2: Implement the smallest focused model blocks (AC 1-3).
  - [x] Add explicit time-major-to-channel-first and channel-first-to-time-major modules; do not hide transposes inside the encoder or add a layout dispatcher.
  - [x] Add a dense no-pooling Conv1d block with positive extents and an odd kernel-size requirement so temporal extent is invariant.
  - [x] Move the legacy `InstanceStandardize` implementation into the focused standardization module, correct it to population statistics and clamp-min epsilon semantics, and support one declared validity channel without inferring padding from values.
  - [x] Compose only with existing `Stages`/`Chain`; add no base class, result/config dataclass, registry, task mode, or new dependency.
- [x] Task 3: Migrate `parkinsons_fog` (AC 3, 4).
  - [x] Stage one explicit real-timestep channel in the consumer-owned store schema and map raw `x`, `y`, and scoring `mask` through `StoredItems`; pad the validity channel with zero.
  - [x] Declare the complete model config as layout -> instance standardization -> dense Conv1d -> inverse layout while preserving convolution construction order.
  - [x] Resolve the same model config in ordinary and calibration paths, record full dataset/model/preprocessing configs in provenance, and keep the masked objective and accumulation guard unchanged.
  - [x] Delete `FogDetector`, `FogWindows`, `fog_windows`, `fog_inputs`, and `normalize_signal`; use the shared dataset/collation path for train, export, evaluation, and inference.
  - [x] Preserve source validation, target/scoring-mask semantics, output names, dynamic axes, prediction ordering, and byte-identical submission formatting.
- [x] Task 4: Govern and measure the components (AC 5).
  - [x] Register the real FoG use for each public block, remove obsolete legacy evidence, remove FoG from stored-dataset and local-model candidates, and regenerate the catalog.
  - [x] Migrate the synthetic supervised fixture's local layout adapter to the DSio adapter as fixture evidence without counting it as a real use.
  - [x] Run admission checks, record owner standing approval, and complete an adversarial review before merge.
  - [x] Measure FoG from the Story 8.2 baseline of 795 non-ingestion lines and repository banned definitions from 26; record the resulting deltas.
- [x] Task 5: Verify the migration (AC 1-5).
  - [x] Run focused signal/preprocessor tests and the complete FoG contract suite, including provenance and MLflow round-trip assertions.
  - [x] Run full pytest, Ruff check/format, mypy, import contracts, catalog/admission, wheel/sdist build, and distribution/consumer-flow contracts.
  - [x] Verify the golden digest remains `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

### Review Findings

- [x] [Review][Patch] Keep the validity channel through the two-convolution block and zero hidden/output padding so real predictions are invariant to co-batched lengths [src/dsio/experimental/model/convolution.py]
- [x] [Review][Patch] Add declarative pre-cast finite/allowed-value constraints and restore FoG target/mask corruption tests [src/dsio/experimental/data/items.py]
- [x] [Review][Patch] Replace tensor-to-Python validity checks with graph-compatible tensor checks [src/dsio/experimental/model/standardization.py]
- [x] [Review][Patch] Accumulate float16/bfloat16 statistics in float32 and keep tiny accepted epsilon values nonzero in the computation dtype [src/dsio/experimental/model/standardization.py]
- [x] [Review][Patch] Version the expanded eight-channel FoG store schema as v2 [reference_projects/kaggle/parkinsons_fog/tasks/data.py]
- [x] [Review][Patch] Reject empty temporal inputs before backend convolution errors [src/dsio/experimental/model/convolution.py]
- [x] [Review][Patch] Compare the configured local Predictor exactly with its MLflow-reloaded form [tests/kaggle_portfolio/test_parkinsons_fog.py]
- [x] [Review][Patch] Add the missing variable-time, inverse-layout, dtype/device, full-graph, and tiny-epsilon proofs [tests/experimental/model/test_signals.py]
- [x] [Review][Patch] Correct the predeclared golden SHA typo in the story record [this file]

## Dev Notes

- `x` remains the sole model input. Padding validity travels as an explicit consumer-declared channel, exactly as the architecture permits; batch `mask` remains the scoring/objective mask and is not sent separately to the model.
- Do not infer padding from zero or another sensor value. A real all-zero timestep is valid data.
- The validity channel is structural, not a learned feature. `InstanceStandardize` validates and preserves it; `DenseConv1d` removes it before learned convolution and, by default, masks hidden/output padding so convolution cannot leak padded context into real boundary predictions. Story 8.7 records FoG's explicit `isolate_padding=False` exception for legacy parity.
- Statistics are per sample and feature over the time axis. Invalid/padded locations are ignored and returned as zero after normalization, reproducing normalization-before-padding while allowing the component to live inside the Predictor's deterministic model path.
- Keep layouts explicit: model preprocessing receives `[B,T,C+1]`, transposes to `[B,C+1,T]`, standardizes while preserving `[B,C+1,T]`, the encoder consumes validity and returns `[B,K,T]`, and the final adapter returns `[B,T,K]`.
- `DenseConv1d` is the proven architecture only: Conv1d -> ReLU -> Conv1d. Do not add depth lists, pooling, residuals, normalization, or activation registries.
- The consumer owns the added store channel and schema version. DSio source and docstrings must not name the consumer.
- New blocks remain experimental despite future second-use plans. Promotion belongs to Story 11.1.

### Previous Story Intelligence

- Story 8.1 made variable-length padding and True-is-valid batch masks explicit. Its finite padding contract remains unchanged; the new validity channel uses representable zero padding.
- Story 8.2 made `MaskedObjective` authoritative for the scoring mask. Do not reuse that mask as padding validity or alter valid-position reduction.
- Story 7.3 established `Stages`/`Chain`, deterministic construction order, and stage-labelled error notes. Reuse them rather than creating another composition.
- Story 7.1 established `StoredItems`; the raw mapping now becomes possible because standardization leaves the local dataset.

### Project Structure Notes

- Put layout adapters in `src/dsio/experimental/model/layout.py`, dense convolution in `src/dsio/experimental/model/convolution.py`, and per-instance standardization beside fixed `Standardize` in `standardization.py`.
- Remove `InstanceStandardize` from the oversized legacy `components.py`; update defining-module evidence and imports together. No compatibility shim is needed during pre-release development.
- Expected consumer changes include FoG components, ingestion, training, downstream tasks, tests, and provenance assertions. Do not touch deployment/orchestration or widen Predictor.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.3]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-1 and constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/component-cohort.md` — blocks #9b, #11, #13]
- [Source: `_bmad-output/implementation-artifacts/8-1-pad-variable-length-batches-with-a-dsio-collator.md`]
- [Source: `_bmad-output/implementation-artifacts/8-2-train-dense-sequence-models-with-masked-objectives.md`]
- [Source: `src/dsio/experimental/model/{components.py,compositions.py,standardization.py}`]
- [Source: `reference_projects/kaggle/parkinsons_fog/components.py`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Red phase: the focused suite failed on missing signal blocks; the review regression suite then reproduced padding-context leakage, lossy mask coercion, and full-graph compilation failure before their patches.
- Review verification: 190 focused admission, data, signal, and FoG tests passed after all three independent review passes were resolved.
- Final full suite: 1,259 passed, 1 deselected.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added explicit time-major/channel-first layout adapters, a two-layer dense Conv1d block, and population-based per-instance standardization as small configurable native modules.
- Kept padding validity inside `x`: standardization preserves it, while the dense block excludes it from learned inputs and masks intermediate/output padding by default, proving the safe mode is invariant to co-batched sequence lengths. Story 8.7 later configured FoG's audited legacy exception explicitly.
- Migrated FoG to full `StoredItems`, `Chain`, `Stages`, model, objective, and collator configurations; deleted its local dataset, normalization, model, and factory definitions; and versioned the eight-channel store schema.
- Extended `StoredItems` with generic pre-cast `finite` and `allowed_values` constraints so boolean conversion cannot hide corrupt targets or masks.
- Review resolved nine deduplicated actionable findings from blind, edge-case, and acceptance passes, including half-precision stability, graph-compatible checks, empty-time validation, and exact local-vs-MLflow predictor equality.
- FoG non-ingestion code fell from 795 to 761 lines (-34); repository banned local definitions fell from 26 to 22 (-4).
- Verification passed: focused signal/data/admission/FoG suites; exact MLflow reload parity; full `1,259 passed, 1 deselected`; Ruff; format; mypy over 118 source files; three import contracts; wheel/sdist build; and distribution/consumer-flow contracts.
- `tests/golden_metrics.json` remained unchanged at SHA-256 `71371a90d59bb9f51c5d2d89c2acb2e6a6b81cce992ba550e2e6caa0362dc530`.

### File List

- `_bmad-output/implementation-artifacts/8-3-compose-dense-signal-models-with-layout-adapters-and-per-instance-standardization.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/candidates.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/conventions.md`
- `docs/component-warehouse/evidence.yaml`
- `reference_projects/kaggle/parkinsons_fog/components.py`
- `reference_projects/kaggle/parkinsons_fog/data.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/data.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/downstream.py`
- `reference_projects/kaggle/parkinsons_fog/tasks/training.py`
- `reference_projects/self_supervised/components.py`
- `reference_projects/self_supervised/tasks/training.py`
- `reference_projects/supervised/components.py`
- `reference_projects/supervised/tasks/training.py`
- `src/dsio/experimental/data/items.py`
- `src/dsio/experimental/model/__init__.py`
- `src/dsio/experimental/model/components.py`
- `src/dsio/experimental/model/convolution.py`
- `src/dsio/experimental/model/layout.py`
- `src/dsio/experimental/model/standardization.py`
- `tests/experimental/data/test_items.py`
- `tests/experimental/model/test_signals.py`
- `tests/kaggle_portfolio/assertions.py`
- `tests/kaggle_portfolio/test_parkinsons_fog.py`
- `tests/model/test_module.py`
- `tests/reference_flows/test_self_supervised_flow.py`
- `tests/reference_flows/test_supervised_flow.py`
- `tools/catalog.py`

## Change Log

- 2026-10-03: Created Story 8.3 with an explicit padding-validity channel and predeclared normalization tolerance.
- 2026-10-03: Added, migrated, reviewed, and verified the shared dense-signal components and padding-safe FoG configuration.
- 2026-10-03: Story 8.7 corrected the real-window reduction-order defect exposed by its zero-tolerance parity run.
