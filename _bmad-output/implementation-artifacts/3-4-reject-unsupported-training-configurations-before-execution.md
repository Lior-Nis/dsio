---
baseline_commit: 9f5bd48
---

# Story 3.4: Reject Unsupported Training Configurations Before Execution

Status: done

## Story

As a training-task author,
I want execution capabilities checked before expensive work begins,
so that an unsupported device, dtype, or operation cannot silently alter experiment semantics.

## Acceptance Criteria

1. Before `Trainer.fit()`, DSIO checks the resolved device and precision plus the assembled model, objective, and accelerator augmentation, and reports every detectable blocking incompatibility.
2. An unavailable operation names the failing component and capability; DSIO never moves it to CPU, changes dtype, or selects a substitute implementation.
3. A supported configuration follows the exact Lightning-resolved execution path and logs the relevant environment and capability provenance to the same MLflow Run.
4. Configurations DSIO cannot verify fail closed and direct novel behavior to the experimental admission path.

## Tasks / Subtasks

- [x] Add one pre-fit capability probe shared by supervised and SSL training (AC: 1-2, 4)
  - [x] Use the already assembled `DsioModule` and a representative real batch rather than a component registry or declared capability model.
  - [x] Exercise model, objective, and training augmentation on Lightning's resolved root device and precision context.
  - [x] Preserve module state and report all independently detectable failures with component names.
- [x] Make the execution path explicit and fail closed (AC: 1-4)
  - [x] Let native Lightning resolve accelerator, devices, and precision once.
  - [x] Reject unsupported multi-device or device/precision paths without fallback.
  - [x] Keep the stable matrix deliberately small and point unverified combinations to experimental admission.
- [x] Record capability evidence through native MLflow (AC: 3)
  - [x] Log requested and resolved execution settings plus relevant runtime versions to the active MLflow Run.
  - [x] Do not introduce an execution-result or environment model parallel to MLflow and existing provenance.
- [x] Prove fail-fast behavior, no fallback, provenance, and existing training compatibility through tests and release gates (AC: 1-4)

## Dev Notes

### Minimal shape

- Add one cohesive `dsio.train.capabilities` module and one call immediately before each `Trainer.fit()`.
- The probe consumes native `Trainer`, `DsioModule`, and batch objects and returns a primitive mapping suitable for MLflow parameters. No registry, capability dataclass, result dataclass, custom device abstraction, or alternate execution path.
- Do not duplicate Lightning's accelerator selection. Inspect the Trainer's resolved strategy and precision plugin.

### Compatibility policy

- Stable support starts with one process on CPU or CUDA using true float32. Those are the execution semantics exercised by DSIO's current component contracts; other devices, distributed strategies, and precision modes enter through the governed experimental admission path.
- `auto` remains a valid request only because Lightning resolves it before the probe; provenance records both requested and resolved values.
- An explicitly requested unavailable accelerator must fail in Lightning construction. DSIO must preserve that failure rather than falling back.

### Operation probe

- Use the first training batch after collation, transferred to the resolved root device, and Lightning's native precision context.
- Execute the configured train-only augmentation and objective without stepping an optimizer. Restore training/evaluation mode and metric state so the proof does not become training.
- Aggregate failures that can be checked independently. Error messages identify the configured concern and end with the experimental admission direction for unverified behavior.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 3.4, FR20]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Training system and component compatibility]
- [Source: `src/dsio/train/trainer.py`]
- [Source: `src/dsio/model/module.py`]
- [Source: `src/dsio/train/tracking.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 3.3 at `9f5bd48`; compared declarative per-component capability metadata with a probe of the real assembled chain and chose the latter to avoid an unverifiable parallel registry.
- 2026-09-22: First independent review blocked the candidate on detached losses, retained lazy/parameter/Python state, incomplete objective validation, unproved optimizer construction, string multi-device selectors, non-Torch RNG leakage, CUDA device-zero handling, and raw transfer failures. Every finding was reproduced and fixed test-first.
- 2026-09-22: Second review found persistent-worker loader initialization changed the real shuffled training order and requested component identities in transfer diagnostics. The probe now samples through a separate one-shot loader, and both closures have regressions.
- 2026-09-22: Hardened candidate passed distribution/consumer contracts (8 tests), the remaining repository suite (872 tests; 3 live tests deselected), Ruff, mypy across 78 source files, and all three import contracts.
- 2026-09-22: Exact candidate `80f6fe5` was approved independently by the acceptance, blind, and edge review layers with AC1-AC4 satisfied and no remaining actionable finding.

### Completion Notes List

- Supervised and SSL runners now call one capability seam after native Trainer construction and before `fit()`.
- Requested CUDA unavailability, multi-device execution, unsupported devices, and non-float32 precision fail closed without fallback and point to experimental admission.
- The real assembled augmentation and model/objective boundary executes forward and backward on a representative batch using Lightning's resolved device and precision context.
- Probe sampling uses a separate non-persistent loader and preserves all seeded RNG streams, so it never initializes or advances the real training loader. Execution uses an isolated module clone so parameters, lazy initialization, caches, buffers, metrics, modes, and arbitrary component state cannot alter the real training module.
- The proof validates the complete objective result, executes backward, and constructs the native optimizer/scheduler before reporting support.
- Requested and resolved execution settings plus PyTorch, CUDA, cuDNN, strategy, and device identity are native MLflow parameters on the training Run.

### File List

- `_bmad-output/implementation-artifacts/3-4-reject-unsupported-training-configurations-before-execution.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `src/dsio/train/capabilities.py`
- `src/dsio/train/ssl_task.py`
- `src/dsio/train/torch_task.py`
- `tests/train/test_capabilities.py`
- `tests/train/test_ssl_runner.py`
- `tests/train/test_torch_runner.py`

### Change Log

- 2026-09-22: Created Story 3.4 and started implementation.
- 2026-09-22: Added the shared fail-closed capability proof and MLflow execution evidence; moved the story to review after the full release gate passed.
- 2026-09-22: Completed Story 3.4 and Epic 3 after the hardened exact candidate passed all release and independent review gates.
