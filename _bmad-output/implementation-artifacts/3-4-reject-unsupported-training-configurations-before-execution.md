---
baseline_commit: 9f5bd48
---

# Story 3.4: Reject Unsupported Training Configurations Before Execution

Status: in-progress

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

- [ ] Add one pre-fit capability probe shared by supervised and SSL training (AC: 1-2, 4)
  - [ ] Use the already assembled `DsioModule` and a representative real batch rather than a component registry or declared capability model.
  - [ ] Exercise model, objective, and training augmentation on Lightning's resolved root device and precision context.
  - [ ] Preserve module state and report all independently detectable failures with component names.
- [ ] Make the execution path explicit and fail closed (AC: 1-4)
  - [ ] Let native Lightning resolve accelerator, devices, and precision once.
  - [ ] Reject unsupported multi-device or device/precision paths without fallback.
  - [ ] Keep the stable matrix deliberately small and point unverified combinations to experimental admission.
- [ ] Record capability evidence through native MLflow (AC: 3)
  - [ ] Log requested and resolved execution settings plus relevant runtime versions to the active MLflow Run.
  - [ ] Do not introduce an execution-result or environment model parallel to MLflow and existing provenance.
- [ ] Prove fail-fast behavior, no fallback, provenance, and existing training compatibility through tests and release gates (AC: 1-4)

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

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/3-4-reject-unsupported-training-configurations-before-execution.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 3.4 and started implementation.
