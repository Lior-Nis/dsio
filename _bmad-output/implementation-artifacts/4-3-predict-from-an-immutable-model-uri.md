---
baseline_commit: 52e9ad1
---

# Story 4.3: Predict from an Immutable Model URI

Status: in-progress

## Story

As an inference consumer,
I want to call `dsio.inference.predict(model_uri, inputs)`,
so that I can obtain validated predictions without knowing the model's training implementation.

## Acceptance Criteria

1. Given an immutable URI for a valid logged predictor and signature-compatible inputs, `predict` loads the packaged deterministic preprocessing, model, normalization, and semantic validation and returns native prediction data rather than a DSIO result wrapper.
2. Batched output ordering and stable sample identity align with the inputs, and declared output fields satisfy the logged MLflow Signature and packaged semantic validator.
3. Incompatible input, mutable or missing model reference, unsupported device request, or invalid output fails at its boundary with an actionable error and without silent schema, device, or dtype coercion.
4. Batch, streaming, windowed, online, serving, scheduling, and deployment lifecycle remain caller-owned.

## Tasks / Subtasks

- [ ] Add one loading-and-prediction seam under `dsio.inference` (AC: 1-4)
  - [ ] Accept only an immutable MLflow Logged Model URI, native signature-compatible inputs, and an explicit supported device.
  - [ ] Load through MLflow PyFunc so callers do not know the training implementation.
  - [ ] Return the model's native prediction mapping without a DSIO result class.
- [ ] Enforce the logged contract without coercion (AC: 2-3)
  - [ ] Validate exact declared input names, tensor dtypes, ranks, and fixed dimensions before inference.
  - [ ] Validate exact declared output fields and tensor contracts after inference.
  - [ ] Require output sample identities to match input order exactly.
- [ ] Fail at explicit evidence and execution boundaries (AC: 3)
  - [ ] Reject mutable Registry aliases/versions, Run artifact URIs, local paths, and malformed or missing Logged Model references.
  - [ ] Reject unsupported device requests rather than remapping silently.
  - [ ] Translate MLflow load, schema, and inference failures into one actionable inference boundary error while preserving the cause.
- [ ] Prove immutable loading, packaged preprocessing/validation, exact schema and identity, failure boundaries, and absence of lifecycle policy through focused and release tests (AC: 1-4)

## Dev Notes

### Minimal shape

- Add `dsio.inference.loading`; keep the public API to `predict` plus one boundary error. Do not add a loader class, cache, request/result dataclass, backend registry, batch runner, server, or deployment configuration.
- Use MLflow's native PyFunc loader and the logged `ModelSignature`; do not duplicate the signature as a DSIO schema model.
- The supported URI is the immutable MLflow 3 Logged Model form `models:/<model_id>`. Registry names, versions, and aliases are mutable policy surfaces and remain out of scope.

### Exact inference contract

- Require mapping inputs containing NumPy arrays that exactly match the named tensor signature. Shape wildcards remain governed by the native MLflow signature; fixed dimensions and dtypes must match exactly before calling the model.
- The packaged PyFunc delegates preprocessing and semantic output validation to the complete `Predictor`. DSIO additionally validates the returned native mapping against the output signature and exact `sample_id` order.
- Current exported PyFunc predictors are CPU representations. Expose the device boundary so any non-CPU request fails explicitly rather than being ignored or remapped.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 4.3, FR32]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Metrics, evaluation, and inference]
- [Source: `src/dsio/inference/export.py`]
- [Source: MLflow 3.16 Logged Model, Model Signature, and PyFunc loading APIs]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 4.2 at `52e9ad1`; selected one native PyFunc loading seam with exact signature checks over any DSIO loading hierarchy.

### Completion Notes List

### File List

- `_bmad-output/implementation-artifacts/4-3-predict-from-an-immutable-model-uri.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`

### Change Log

- 2026-09-22: Created Story 4.3 and started implementation.
