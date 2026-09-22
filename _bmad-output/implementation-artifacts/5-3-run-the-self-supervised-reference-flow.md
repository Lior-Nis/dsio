---
baseline_commit: af1bfaf
---

# Story 5.3: Run the Self-Supervised Reference Flow

Status: done

## Story

As a new DSio consumer,
I want an executable self-supervised example using the same public training spine,
so that I can verify DSio supports a distinct training paradigm without a second framework.

## Acceptance Criteria

1. In a clean supported environment with local MLflow tracking, an ordinary project-owned Prefect flow trains through the exact `DsioModule` and `DsioDataModule` with named SSL components and no SSL-specific runner or subclass.
2. Multi-view stochastic augmentation runs inside `training_step()` after Lightning device transfer, preserves source sample identity across declared views, and reproduces views and training outputs from the same declared seed material.
3. The completed flow links data, split, component, seed, training, evaluation, model, and environment provenance through native MLflow records; its exported predictor produces signature-valid, semantically valid output.
4. Compared with the supervised reference project, orchestration, tracking, data, Lightning, evaluation, and inference use the same public contracts; variation is confined to named components and serializable configuration.

## Tasks / Subtasks

- [x] Add a consumer-owned self-supervised reference package (AC: 1, 4)
  - [x] Reuse the supervised reference project's ordinary data, split, evaluation, and inference tasks.
  - [x] Define only the named embedding model, contrastive objective, and export normalizer needed by SSL.
- [x] Execute the SSL path through the exact reusable classes (AC: 1-2)
  - [x] Instantiate `DsioModule` and `DsioDataModule` directly.
  - [x] Inject `TwoView` accelerator-side augmentation and a named contrastive objective.
  - [x] Record every behavior-changing component and parameter in execution provenance.
- [x] Export, evaluate, and infer through the shared public contracts (AC: 3-4)
  - [x] Build and log an immutable predictor from successful checkpoint evidence.
  - [x] Reuse the same downstream MLflow evaluation and inference tasks as supervised training.
- [x] Prove deterministic views, outputs, identity preservation, lineage, and installed-wheel execution in CI (AC: 1-4)

## Dev Notes

### Minimal shape

- This is a second consumer project, not a new DSio task kind. Do not use the legacy SSL runner or introduce a framework wrapper.
- Reuse the supervised reference tasks wherever their inputs and outputs already fit. Duplication is justified only for SSL-specific training and export.
- Keep scientific interpretation out of the integration smoke metric: the held-out scalar embedding probe demonstrates the evaluation contract, not model quality or promotion readiness.
- Test the augmentation seam at the observable batch boundary: device, source IDs, view IDs, repeatability, and downstream outcome.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 5.3, FR38]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — First generic acceptance slice]
- [Source: `reference_projects/supervised/`]
- [Source: `src/dsio/train/augmentation.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 5.2 at `af1bfaf`; chose shared consumer tasks plus SSL-only named components over a parallel flow framework.
- 2026-09-22: Focused tests passed for exact class reuse, in-step deterministic views, stable outcomes, MLflow lineage, and both reference projects.
- 2026-09-22: Installed-wheel probe passed with supervised and self-supervised consumer projects copied outside the checkout package.
- 2026-09-22: Full gate passed: 1047 tests, Ruff, mypy, import contracts, and wheel/sdist build.
- 2026-09-22: Three-layer review exposed a time/channel layout mismatch that made jitter a no-op, plus split-name and resolved-environment identity gaps. Added regressions and corrected all three.
- 2026-09-22: Post-fix full gate passed: 1048 tests, Ruff, mypy, import contracts, wheel/sdist build, and copied-consumer wheel execution.
- 2026-09-22: Acceptance, blind adversarial, and edge-case reviews all passed exact commit `b8429d3` with no findings.

### Completion Notes List

- Added one ordinary Prefect SSL flow that reuses the supervised data, split, evaluation, and inference tasks unchanged.
- Added only three project-specific components: a tiny embedding model, NT-Xent objective adapter, and semantically validated embedding-norm output.
- Multi-view augmentation remains the existing `TwoView` injected into the exact `DsioModule`; the test observes it only while `training_step()` is active and proves device, source/view identity, and deterministic tensors.
- Training identity includes all named components, augmentation identity, seed, optimizer, objective, loader, Trainer settings, and Lightning's resolved execution environment; the same evidence is logged as native MLflow parameters.
- The SSL dataset factory presents the store's time-major arrays as channel-first tensors, and the integration test proves both views differ from the source and from each other while replaying exactly.

### File List

- `_bmad-output/implementation-artifacts/5-3-run-the-self-supervised-reference-flow.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `reference_projects/self_supervised/__init__.py`
- `reference_projects/self_supervised/components.py`
- `reference_projects/self_supervised/flow.py`
- `reference_projects/self_supervised/tasks/__init__.py`
- `reference_projects/self_supervised/tasks/export.py`
- `reference_projects/self_supervised/tasks/training.py`
- `reference_projects/supervised/tasks/data.py`
- `src/dsio/train/capabilities.py`
- `tests/reference_flows/conftest.py`
- `tests/reference_flows/test_self_supervised_flow.py`
- `tests/reference_flows/test_supervised_flow.py`
- `tests/train/test_capabilities.py`
- `tests/test_built_distribution.py`

### Change Log

- 2026-09-22: Created Story 5.3 and started implementation.
- 2026-09-22: Implemented and validated the self-supervised reference flow; moved to review.
- 2026-09-22: Fixed all initial adversarial-review findings and split the growing SSL task module by responsibility.
- 2026-09-22: Completed all review and release gates; marked done.
