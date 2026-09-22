---
baseline_commit: e953e82
---

# Story 3.1: Train a Task Through the Exact DSio Lightning Classes

Status: done

## Story

As a training-task author,
I want to train through the concrete `DsioModule` and `DsioDataModule` classes,
so that every project exercises one battle-tested training path.

## Acceptance Criteria

1. Given an ordinary PyTorch model, valid objective, and configured `DsioDataModule`, a native Lightning `Trainer.fit()` executes training and validation through exact instances of both DSio classes without a DSio runner or project subclass.
2. The objective is one callable receiving `(model, batch, stage)` and returning a mapping with mandatory scalar tensor `loss` plus optional named scalar tensor metrics; malformed results fail at the objective boundary with an actionable error.
3. `DsioModule` owns the shared train/validation/test step and emits every objective value through `LightningModule.log()`; it contains no project/task/paradigm dispatch.
4. Defining a subclass of either DSio Lightning class fails immediately and directs variation through injected components.
5. Native Lightning checkpoints preserve model, objective-module, optimizer, and loop state and can resume the same instantiated training configuration; checkpoints remain training artifacts, not predictors.
6. Existing component-chain consumers use the same exact `DsioModule` through ordinary model/objective components, and root `import dsio` remains inert.

## Tasks / Subtasks

- [x] Make `DsioModule` the generic Lightning composition root (AC: 1-3)
  - [x] Accept one `nn.Module` model and one objective callable.
  - [x] Validate the batch identity and flat objective-result contract at the narrowest boundary.
  - [x] Share one step implementation and log loss/metrics through Lightning.
- [x] Preserve proven component-chain behavior without embedding it in the training class (AC: 3, 6)
  - [x] Move the existing encoder/head chain and loss diagnostics into ordinary reusable components.
  - [x] Adapt current DSio consumers to instantiate the generic class.
- [x] Enforce and verify the exact-class path (AC: 1, 4-6)
  - [x] Reject subclass definitions for both Lightning classes.
  - [x] Cover direct native fit/validation and native checkpoint resume with exact class assertions.
  - [x] Cover malformed batches/objective results and retained existing behavior.
- [x] Complete the full quality and independent review gates before merge (AC: 1-6)

## Dev Notes

### Minimal shape

- `DsioModule` is a composition root, not a component chain or mode dispatcher. It knows only the model, objective, and the temporary native AdamW defaults that Story 3.2 will replace with injected optimizer/scheduler factories.
- Use one flat mapping (`loss` plus metric names) rather than an objective-result dataclass or parallel metrics model.
- Keep the proven encoder/head chain only as an ordinary reusable model component needed by existing consumers. Its loss adapter is an ordinary objective component.
- Direct `Trainer.fit(module, datamodule=data)` is the public training path. Do not add a DSio runner, trainer wrapper, or lifecycle result.

### Checkpoint boundary

- Resume is native Lightning behavior: instantiate the same configuration and pass `ckpt_path` to `Trainer.fit()`.
- The checkpoint owns training/optimizer/loop state. It is deliberately not the predictor artifact described by Epic 4.
- Importable component configuration and provenance are Story 3.2; accelerator augmentation is Story 3.3.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 3.1, FR12, FR14]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Training system]
- [Source: `src/dsio/model/module.py`]
- [Source: `src/dsio/data/loading/module.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 2.5 at `e953e82`; reconciled the existing hard-coded chain with the approved generic objective contract and native Lightning resume semantics.
- 2026-09-22: Replaced the hard-coded training class chain with one model/objective contract; retained the proven chain and loss diagnostics as ordinary `ComponentChain` and `LossObjective` components.
- 2026-09-22: Added stable sample identity to the predecessor window batches so every current training consumer enters the same identity-bearing module boundary.
- 2026-09-22: Candidate release gate passed: 823 tests passed (3 deselected), locked sync, source/wheel builds, Ruff, mypy, all import contracts, and diff checks.
- 2026-09-22: Blind review found two window-ID encodings; identity now has one implementation on `WindowIndex`, shared by governed examples and full or reordered datasets.
- 2026-09-22: Edge review closed non-finite optimizer values, Lightning loss-name collisions, diagnostic loss replacement, and prediction type/cardinality mismatches at their narrow boundaries.
- 2026-09-22: Edge re-review added fail-closed validation for optional prediction rows, malformed window entity codes, and oversized numeric hyperparameters.
- 2026-09-22: Final edge pass moved entity-code validation before narrowing conversion and required legacy prediction rows to be a flat vector.
- 2026-09-22: Exact-SHA edge review closed the same narrowing gap for window starts and rejected non-integral prediction row coordinates.
- 2026-09-22: Final coordinate audit bounded complete windows to int64 and restricted legacy rows to dense ordinary integer tensors.

### Completion Notes List

- Native `Trainer.fit(module, datamodule=data)` now trains exact DSio classes directly; no DSio runner, result model, or subclass is needed.
- One flat objective result carries scalar tensor `loss` and optional scalar tensor metrics, all emitted through `self.log()` by the shared step.
- Model and stateful objective modules participate in native checkpoint state; a fresh same-config instance resumes optimizer and loop progress through `ckpt_path`.
- Existing supervised, self-supervised, token, and callback paths use the same generic module through ordinary components rather than a compatibility constructor branch.
- Both Lightning composition roots reject subclass definitions and direct variation toward injected components.
- Exact commit `0a42f27` passed blind, edge-case, and acceptance review plus the full release gate: 858 passed, 3 deselected, locked dependencies, Ruff, mypy, build, and diff checks.

### File List

- `_bmad-output/implementation-artifacts/3-1-train-a-task-through-the-exact-dsio-lightning-classes.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/batches.py`
- `src/dsio/data/adapters.py`
- `src/dsio/data/loading/module.py`
- `src/dsio/data/views.py`
- `src/dsio/dataset/dataset.py`
- `src/dsio/model/chain.py`
- `src/dsio/model/module.py`
- `src/dsio/train/ssl_task.py`
- `src/dsio/train/torch_task.py`
- `tests/dataset/test_dataset.py`
- `tests/data/test_store.py`
- `tests/model/test_components.py`
- `tests/model/test_module.py`
- `tests/model/test_token_corpus.py`
- `tests/model/test_training_spine.py`
- `tests/train/test_callbacks.py`
- `tests/train/test_torch_runner.py`

### Change Log

- 2026-09-22: Created Story 3.1 and started implementation.
- 2026-09-22: Implemented the exact native-Lightning training path and passed the complete local quality gate.
- 2026-09-22: Closed independent review findings, approved exact release candidate `0a42f27`, and completed Story 3.1.
