---
baseline_commit: af1bfaf
---

# Story 5.3: Run the Self-Supervised Reference Flow

Status: in-progress

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

- [ ] Add a consumer-owned self-supervised reference package (AC: 1, 4)
  - [ ] Reuse the supervised reference project's ordinary data, split, evaluation, and inference tasks.
  - [ ] Define only the named embedding model, contrastive objective, and export normalizer needed by SSL.
- [ ] Execute the SSL path through the exact reusable classes (AC: 1-2)
  - [ ] Instantiate `DsioModule` and `DsioDataModule` directly.
  - [ ] Inject `TwoView` accelerator-side augmentation and a named contrastive objective.
  - [ ] Record every behavior-changing component and parameter in execution provenance.
- [ ] Export, evaluate, and infer through the shared public contracts (AC: 3-4)
  - [ ] Build and log an immutable predictor from successful checkpoint evidence.
  - [ ] Reuse the same downstream MLflow evaluation and inference tasks as supervised training.
- [ ] Prove deterministic views, outputs, identity preservation, lineage, and installed-wheel execution in CI (AC: 1-4)

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

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/5-3-run-the-self-supervised-reference-flow.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 5.3 and started implementation.
