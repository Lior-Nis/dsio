---
baseline_commit: 4902f67
---

# Story 4.1: Build a Self-Contained Predictor from Training Evidence

Status: in-progress

## Story

As a model author,
I want to turn successful training evidence into a predictor with its required preprocessing,
so that inference does not depend on reconstructing project training code by convention.

## Acceptance Criteria

1. Given a digest-pinned checkpoint on a successful MLflow Run and importable components, predictor construction packages model state, deterministic preprocessing, output normalization, and semantic validation while excluding stochastic training-only augmentation.
2. Local prediction applies preprocessing, model forward, normalization, and semantic validation in order and preserves each input `sample_id` plus declared output fields.
3. Non-importable, non-serializable, incompatible, or invalid predictor components fail during construction before any predictor is logged, naming the unsupported boundary.
4. The resumable Lightning checkpoint remains distinct from the completed inference predictor; neither creates a DSIO model-registry entity.

## Tasks / Subtasks

- [ ] Add the `dsio.inference` package and one concrete predictor (AC: 1-2)
  - [ ] Compose native deterministic preprocessing, native model inference, output normalization, and semantic validation behind one `nn.Module`.
  - [ ] Accept identity-bearing tensor batches and return a native mapping with aligned `sample_id` values.
  - [ ] Keep stochastic training augmentation and the Lightning training system outside the predictor.
- [ ] Build from immutable successful checkpoint evidence (AC: 1, 4)
  - [ ] Verify the MLflow Run completed successfully and the existing digest-pinned artifact bytes still match.
  - [ ] Load only the trained model state from the Lightning checkpoint, strictly, on CPU.
  - [ ] Retain the immutable checkpoint URI and digest as predictor provenance without copying or registering the checkpoint.
- [ ] Fail construction at reproducibility boundaries (AC: 3)
  - [ ] Require named importable model, preprocessing, normalization, and validator behavior.
  - [ ] Prove the completed predictor can be serialized before returning it.
  - [ ] Name state, importability, serialization, and semantic contract failures directly.
- [ ] Prove local order, identity, contract, and artifact separation through tests and full release gates (AC: 1-4)

## Dev Notes

### Minimal shape

- Create `dsio.inference` as a package because Stories 4.1-4.3 add distinct predictor, MLflow logging, and loading responsibilities.
- Start with `predictor.py`; do not add an inference result class, model registry, schema hierarchy, exporter registry, or serving abstraction.
- `Predictor` is an ordinary native `nn.Module`. Normalization is an injected `nn.Module`; semantic validation is an importable callable returning `None` on success.
- Return a plain mapping of native tensors/sequences. `sample_id` is reserved and copied from input, never manufactured by a normalizer.

### Evidence and state

- Reuse the existing `ArtifactRef` digest check. Its MLflow Run must be `FINISHED` before construction.
- A Lightning checkpoint remains the resume artifact. Extract only `state_dict` keys below `model.` into a cloned inference model and load strictly.
- Store only checkpoint URI and digest strings on the predictor as provenance. Story 4.2 will log the predictor through native MLflow model facilities.

### Reproducibility

- Reject a whole `DsioModule`; callers pass its deterministic `model` only, which structurally excludes objective, optimizer, and training augmentation.
- Validate every injected behavior is importable and the assembled predictor is serializable. Do not attempt to infer deployment topology or silently substitute devices/dtypes.
- The predictor runs in evaluation/inference mode and does not mutate caller inputs.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 4.1, FR16, FR19, FR33]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Metrics, evaluation, and inference]
- [Source: `src/dsio/model/module.py`]
- [Source: `src/dsio/train/artifacts.py`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 3.4 at `4902f67`; compared rebuilding a training task, serializing a whole `DsioModule`, and loading only `model.` checkpoint state into an inference composition. Selected the last to keep resume and inference artifacts structurally distinct.

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/4-1-build-a-self-contained-predictor-from-training-evidence.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 4.1 and started implementation.
