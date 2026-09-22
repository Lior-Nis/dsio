---
title: 'Harden the self-supervised reference flow'
type: 'bugfix'
created: '2026-09-22'
status: 'done'
baseline_commit: '6d58b985777b1d6d276b2dda5fd2bbe188d3d93d'
context:
  - 'docs/adr/0015-lightning-is-the-only-training-path.md'
  - 'docs/adr/0019-versioned-library-with-project-owned-prefect-flows.md'
  - 'docs/superpowers/specs/2026-09-18-generic-experiment-spine.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The SSL reference trains on channel-first tensors but exports a Predictor that feeds the model time-major tensors; its augmentation and Trainer settings also have duplicated sources of truth, and its DataModule cannot discard an undersized contrastive batch.

**Approach:** Use the existing Predictor preprocessor seam for deterministic layout conversion and shape validation, make native Trainer and augmentation configuration single-source, and add one train-only `drop_last` option to the canonical loading interface.

## Boundaries & Constraints

**Always:** Keep `DsioModule` and `DsioDataModule` as the exact training classes; keep stochastic augmentation exclusively inside `training_step()`; bind requested settings and resolved execution behavior into Execution Identity; preserve existing defaults for all loading callers; use native Lightning, MLflow, Prefect, and PyTorch interfaces.

**Ask First:** Any change that moves project-owned Prefect tasks into DSIO, changes the Predictor's external time-major input contract, or changes existing loader defaults.

**Never:** Add an SSL runner, result model, task dispatcher, layout mode flag, or project registry; call `check_training_capabilities` or `representative_batch` from this reference task; introduce a new configuration dataclass; modify the four repository-root review documents.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| SSL inference | `[batch, time, channels]` Predictor input with declared time/channel extents | deterministic contiguous `[batch, channels, time]` model input equal to the training dataset tensor | wrong rank or axis extent fails with a clear Predictor error |
| Contrastive loading | train membership leaves a partial batch | train drops only the partial batch; observation phases always retain every assignment | invalid phase flags or fewer than one full train batch fail during DataModule setup |
| Existing loading | no `drop_last` option | every assigned sample remains observable | no behavior change |
| Training configuration | one `_TRAINER` and configured augmentation component graph | shared builders construct runtime behavior and the same values enter provenance | unsupported requested capability fails before Trainer construction |

</frozen-after-approval>

## Code Map

- `reference_projects/self_supervised/components.py` -- project-owned input-layout adapter and embedding contract.
- `reference_projects/self_supervised/tasks/export.py` -- Predictor assembly and component provenance.
- `reference_projects/self_supervised/tasks/training.py` -- native Trainer, augmentation, DataModule, and Execution Identity composition.
- `src/dsio/data/loading/loaders.py` -- deterministic native DataLoader construction.
- `src/dsio/data/loading/module.py` -- phase-to-loader configuration seam.
- `src/dsio/train/trainer.py` -- single TrainerConfig-to-Lightning construction path.
- `src/dsio/train/capabilities.py` -- requested and resolved execution evidence.
- `tests/data/loading/test_data_module.py` -- public loading-interface behavior.
- `tests/train/test_trainer.py` -- exact Trainer configuration mapping.
- `tests/reference_flows/test_self_supervised_flow.py` -- executable end-to-end contract.

## Tasks & Acceptance

**Execution:**
- [x] `tests/reference_flows/test_self_supervised_flow.py`, `reference_projects/self_supervised/{components.py,tasks/export.py}` -- first prove the current layout bug with non-square multi-channel values; then export an importable, parameterless-layout adapter whose declared channel/time extents validate the external contract. Assert exact transposition, contiguity, input immutability, rank/extent errors, and value equality with `UnlabelledSamples`; record its component configuration and `split_digest` in export provenance.
- [x] `tests/data/loading/test_data_module.py`, `src/dsio/data/loading/{loaders.py,module.py}` -- specify and implement strict phase-aware `drop_last` with false defaults, rejecting `True` for validate/test/predict and rejecting a dropped train phase with no full batch. Cover direct `build_loader`, DataModule defaults, validation, retained Sample Identities, and deterministic shuffled truncation. This deliberately narrows Story 2.5's exact-membership rule only for fixed-composition training batches.
- [x] `tests/train/test_trainer.py`, `src/dsio/train/{trainer.py,capabilities.py}` -- add `limit_val_batches: int | float | None = None`, reject booleans, validate native integer/fraction ranges, preserve its type, and forward it exactly. Record the builder's effective deterministic mode (`warn` or `false`) in resolved capability evidence.
- [x] `tests/reference_flows/test_self_supervised_flow.py`, `reference_projects/self_supervised/tasks/training.py` -- declare exactly one `_TRAINER`, moving the removed flat keys into it with their current values (`max_epochs=3`, `accelerator="auto"`, `devices=1`, `deterministic=True`, `checkpoint=False`, `log_every_n_steps=1`, `limit_val_batches=0`). Use `check_requested_capabilities(_TRAINER)`, `build_callbacks(..., has_validation=False)`, `build_trainer(...)`, then `resolve_training_capabilities(...)`; `limit_val_batches=0` is load-bearing because the task maps no validate role. Accept the shared builder's `deterministic="warn"` policy explicitly. Construct both `TwoView` and its augmentor with `resolve_component` from the same component mappings recorded as `module.augmentation_identity`; explicitly record and pass shuffle/drop-last mappings, with train-only `drop_last=True`.
- [x] Run focused and full distribution, test, lint, typing, and import-contract gates.

**Acceptance Criteria:**
- Given raw store-shaped Predictor inputs, when the SSL model is exported, evaluated, or invoked, then one recorded deterministic preprocessor validates the external extents and supplies values identical to the channel-first training tensor.
- Given any mapped DataModule phase, when `drop_last` is omitted or false, then existing behavior is unchanged; train-only true discards one remainder but cannot yield zero batches, and observation-phase true is rejected.
- Given SSL training configuration, when the task constructs its Trainer, callbacks, DataModule, and augmentation, then the same serializable values drive runtime behavior, Execution Identity, and MLflow evidence with no inline second builder.
- Given an unsupported requested training capability, when the SSL task begins, then `check_requested_capabilities` fails before Trainer construction; after construction, `resolve_training_capabilities` records the resolved path without executing augmentation outside `training_step()`.

## Spec Change Log

- 2026-09-22: Design review pinned shape-sensitive layout proof, train-only/nonempty `drop_last`, exact native Trainer construction, configured-component augmentation, and complete execution identity.
- 2026-09-22: Implemented all approved tasks test-first; 1078 broad tests and eight installed-distribution tests pass with static and import gates clean.
- 2026-09-22: Adversarial review corrected resolved determinism to inspect PyTorch's active state and hardened adapter extents, observation-phase truncation invariants, and test isolation.
- 2026-09-22: Claude Code review centralized the signal-shape contract and drop-last validation, normalized configured view identities, and removed a recomputed path; two suggestions were rejected because runtime state and the existing end-to-end shape guard provide stronger proof.

## Design Notes

The Predictor already owns deterministic preprocessing that must run identically for evaluation and inference. A project-owned `nn.Module` adapter therefore fixes layout once without parameterizing shared Prefect tasks; fixed extents make axis meaning testable without introducing a layout mode. `drop_last` belongs beside phase-aware shuffle because DSIO owns native DataLoader construction, but only training may discard assignments. The consuming task records complete shuffle and drop-last mappings, including false values, so defaults cannot hide behavior.

`build_trainer` intentionally maps requested deterministic mode to Lightning's canonical `"warn"` policy; resolved capability evidence records that effective mode. The supervised reference's inline Trainer construction, the older `dsio.dataset.dataset.make_loader` seam, and the opposing supervised/SSL internal layouts are named follow-ups, not changes in this bugfix.

## Verification

**Commands:**
- `uv run pytest -q tests/data/loading/test_data_module.py tests/train/test_trainer.py tests/reference_flows/test_self_supervised_flow.py tests/reference_flows/test_supervised_flow.py` -- focused contracts pass.
- `uv build && uv run pytest -q tests/test_distribution.py tests/test_built_distribution.py tests/test_project_flow.py` -- installed consumers pass.
- `uv run --group benchmark pytest -q --ignore=tests/test_distribution.py --ignore=tests/test_built_distribution.py --ignore=tests/test_project_flow.py` -- full suite passes with the declared benchmark dependencies.
- `uv run ruff check . && uv run mypy && uv run lint-imports && git diff --check` -- static and import gates pass.

## Suggested Review Order

**Reference flow composition**

- Start with the single-source runtime configuration and native Lightning assembly.
  [`training.py:34`](../../reference_projects/self_supervised/tasks/training.py#L34)

- Follow configuration through DataModule, augmentation, provenance, and training.
  [`training.py:81`](../../reference_projects/self_supervised/tasks/training.py#L81)

**Inference layout contract**

- Validate and convert external time-major tensors at the Predictor boundary.
  [`components.py:43`](../../reference_projects/self_supervised/components.py#L43)

- Keep model shape and exported preprocessor configuration sourced together.
  [`export.py:32`](../../reference_projects/self_supervised/tasks/export.py#L32)

**Batch composition**

- Enforce train-only truncation while preserving observation-phase membership.
  [`module.py:40`](../../src/dsio/data/loading/module.py#L40)

- Forward validated truncation directly into native PyTorch loading.
  [`loaders.py:20`](../../src/dsio/data/loading/loaders.py#L20)

**Trainer evidence**

- Preserve Lightning's integer-versus-fraction validation and exact forwarding.
  [`trainer.py:16`](../../src/dsio/train/trainer.py#L16)

- Record PyTorch's active deterministic mode as resolved execution evidence.
  [`capabilities.py:224`](../../src/dsio/train/capabilities.py#L224)

**Regression contracts**

- Prove layout conversion with non-square multichannel values and exact identities.
  [`test_self_supervised_flow.py:21`](../../tests/reference_flows/test_self_supervised_flow.py#L21)

- Prove deterministic truncation and rejection of lossy observation loaders.
  [`test_data_module.py:509`](../../tests/data/loading/test_data_module.py#L509)

- Prove exact Trainer mapping and native limit value types.
  [`test_trainer.py:52`](../../tests/train/test_trainer.py#L52)
