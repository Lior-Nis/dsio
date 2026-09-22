---
baseline_commit: fddc018
---

# Story 3.2: Configure Training with Native Importable Components

Status: done

## Story

As a training-task author,
I want to inject named PyTorch and Lightning components into the reusable module,
so that tasks vary without forks, subclasses, or a custom plugin framework.

## Acceptance Criteria

1. A plain component configuration containing a `module:qualname` reference and canonical parameters resolves the named object, validates its expected native contract, and can be recorded unchanged in execution provenance.
2. Anonymous lambdas, closures, local-only objects, unresolved references, and non-canonical parameters fail before training with an actionable named-import requirement.
3. `DsioModule` accepts named native optimizer and optional scheduler factories plus their parameters; `configure_optimizers()` returns only Lightning-supported native structures, with no DSio optimizer or scheduler wrapper.
4. Native Lightning callbacks remain `Trainer` inputs, while objective and TorchMetrics values continue to be emitted exclusively through `LightningModule.log()`.
5. Component references and meaningful parameters are part of execution identity and the native MLflow `provenance.json`; changing either changes the identity.
6. Existing supervised, self-supervised, token, callback, checkpoint, and root-import contracts remain intact.

## Tasks / Subtasks

- [x] Add one generic named-component boundary (AC: 1, 2)
  - [x] Represent configuration as a plain canonical mapping, not a registry or result model.
  - [x] Resolve `module:qualname`, validate import identity and the requested native type, and instantiate with declared parameters.
  - [x] Reject ambiguous or non-importable callables before execution.
- [x] Inject native optimization components into `DsioModule` (AC: 2-4)
  - [x] Replace built-in learning-rate fields with an importable optimizer factory and parameter mapping.
  - [x] Support an optional importable scheduler factory and return native Lightning configuration.
  - [x] Keep callbacks Trainer-owned and metrics on the existing `self.log()` path.
- [x] Make full component configuration first-class provenance (AC: 1, 5)
  - [x] Accept canonical component mappings in execution identity and MLflow provenance.
  - [x] Prove reference and parameter changes alter identity and are logged exactly.
- [x] Migrate current consumers and pass full quality and independent review gates (AC: 1-6)

## Dev Notes

### Minimal shape

- Use one plain mapping: `{"reference": "package.module:qualname", "parameters": {...}}`.
- Add no runtime registry, discovery mechanism, plugin base class, component result object, optimizer wrapper, scheduler wrapper, Trainer wrapper, or MLflow metrics bridge.
- Resolution is generic; model, objective, transform, metric, optimizer, and scheduler variation use the same import boundary and native ecosystem contracts.
- A resolved instance may be stateful, but its identity comes from the named reference and canonical parameters supplied before construction.

### Native ownership

- `DsioModule` remains the composition root and owns `configure_optimizers()` only because Lightning does.
- Optimizers and schedulers remain native PyTorch objects. Scheduler mappings, when needed, are ordinary Lightning-supported mappings.
- Callbacks are passed directly to `lightning.Trainer`; objective outputs remain the one logging boundary.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 3.2, FR13, FR15, FR29]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Execution identity and Training system]
- [Source: `src/dsio/model/module.py`]
- [Source: `src/dsio/tracking/provenance.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 3.1 at `fddc018`; constrained the design to one import resolver, plain mappings, native Lightning/PyTorch structures, and existing provenance.
- 2026-09-22: Removed the model-component and mask registries; current supervised, SSL, token, label, optimizer, and scheduler consumers now use ordinary import references.
- 2026-09-22: Final release gate passed on `7af0ed2`: 8 distribution/consumer-flow tests and 874 remaining tests passed (3 live tests deselected), with locked sync, builds, Ruff, mypy, import contracts, and diff checks clean.
- 2026-09-22: Native TorchMetrics values are accepted as optional objective metrics and passed directly to `LightningModule.log`; loss remains a scalar tensor.
- 2026-09-22: Blind, edge-case, and acceptance reviewers approved exact candidate `7af0ed2` after provenance, scheduler, and lazy TorchMetrics boundary fixes.

### Completion Notes List

- One `module:qualname` resolver constructs and validates native components from canonical plain mappings; no plugin base, registry, or resolution result model was added.
- `DsioModule` validates importability before training and returns native optimizer or Lightning optimizer/scheduler structures.
- Full component mappings are hashed and stored unchanged in MLflow provenance; references and parameters independently affect identity.
- Existing training configs now name models, objectives, transforms, masks, labels, optimizers, and schedulers by import path.
- Secret and ephemeral component parameters are removed before access or canonical validation; scheduler mappings and registered TorchMetrics are validated against Lightning's native runtime contracts.

### File List

- `_bmad-output/implementation-artifacts/3-2-configure-training-with-native-importable-components.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/config/__init__.py`
- `src/dsio/config/components.py`
- `src/dsio/config/registry.py`
- `src/dsio/data/labels.py`
- `src/dsio/model/__init__.py`
- `src/dsio/model/components.py`
- `src/dsio/model/masking.py`
- `src/dsio/model/module.py`
- `src/dsio/model/registry.py` (deleted)
- `src/dsio/tracking/provenance.py`
- `src/dsio/train/assembly.py`
- `src/dsio/train/ssl_task.py`
- `src/dsio/train/torch_task.py`
- `tests/config/test_component_resolution.py`
- `tests/conftest.py`
- `tests/model/test_components.py`
- `tests/model/test_masking.py`
- `tests/model/test_module.py`
- `tests/model/test_registry_bootstrap.py` (deleted)
- `tests/model/test_training_spine.py`
- `tests/tracking/provenance/test_identity.py`
- `tests/tracking/provenance/test_mlflow.py`
- `tests/train/test_execute_seeding.py`
- `tests/train/test_ssl_runner.py`
- `tests/train/test_token_run.py`
- `tests/train/test_torch_runner.py`

### Change Log

- 2026-09-22: Created Story 3.2 and started implementation.
- 2026-09-22: Implemented native importable component configuration and passed the complete candidate gate.
- 2026-09-22: Closed all independent-review findings and completed Story 3.2 on exact candidate `7af0ed2`.
