---
baseline_commit: e953e82
---

# Story 3.1: Train a Task Through the Exact DSio Lightning Classes

Status: in-progress

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

- [ ] Make `DsioModule` the generic Lightning composition root (AC: 1-3)
  - [ ] Accept one `nn.Module` model and one objective callable.
  - [ ] Validate the batch identity and flat objective-result contract at the narrowest boundary.
  - [ ] Share one step implementation and log loss/metrics through Lightning.
- [ ] Preserve proven component-chain behavior without embedding it in the training class (AC: 3, 6)
  - [ ] Move the existing encoder/head chain and loss diagnostics into ordinary reusable components.
  - [ ] Adapt current DSio consumers to instantiate the generic class.
- [ ] Enforce and verify the exact-class path (AC: 1, 4-6)
  - [ ] Reject subclass definitions for both Lightning classes.
  - [ ] Cover direct native fit/validation and native checkpoint resume with exact class assertions.
  - [ ] Cover malformed batches/objective results and retained existing behavior.
- [ ] Complete the full quality and independent review gates before merge (AC: 1-6)

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

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/3-1-train-a-task-through-the-exact-dsio-lightning-classes.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 3.1 and started implementation.
