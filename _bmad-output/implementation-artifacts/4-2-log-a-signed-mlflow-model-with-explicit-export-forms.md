---
baseline_commit: 902e8a3
---

# Story 4.2: Log a Signed MLflow Model with Explicit Export Forms

Status: done

## Story

As a model author,
I want the predictor logged with MLflow's native model contract,
so that its accepted inputs, produced outputs, and supported representations are portable and inspectable.

## Acceptance Criteria

1. Logging a valid predictor with a representative input records an MLflow Model Signature, input example, immutable model URI, source Run, and artifact location for every declared export form.
2. The predictor's semantic validator remains inside every logged representation, so semantically invalid output fails inference rather than being returned as a successful prediction.
3. Declared native PyTorch and PyFunc forms use MLflow's native flavor APIs, and loading either form produces contract-equivalent predictions for the representative fixture.
4. Undeclared forms produce no artifact; unknown or incompatible explicitly requested forms fail before flavor logging and identify the affected form.

## Tasks / Subtasks

- [x] Add one MLflow export seam under `dsio.inference` (AC: 1-4)
  - [x] Accept an existing Predictor, one representative identity-bearing batch, an active Run id, and explicit export-form names.
  - [x] Infer one canonical MLflow signature from the validated input/output fixture.
  - [x] Return native MLflow `ModelInfo` objects rather than a DSIO export result model.
- [x] Log only declared native forms (AC: 2-4)
  - [x] Log the complete predictor through `mlflow.pytorch.log_model` for the native form.
  - [x] Use one minimal `PythonModel` adapter for MLflow-compatible arrays and structured predictor input/output.
  - [x] Keep the semantic validator inside both serialized predictors and create no registered-model entity.
- [x] Fail before logging at explicit export boundaries (AC: 4)
  - [x] Reject empty, duplicate, unknown, or component-incompatible form declarations before the first flavor call.
  - [x] Validate the representative fixture and signature conversion before creating model artifacts.
  - [x] Do not synthesize a fallback form when a requested form cannot be represented.
- [x] Prove signatures, examples, immutable references, form selection, semantic validation, and native round trips through tests and release gates (AC: 1-4)

## Dev Notes

### Minimal shape

- Add `dsio.inference.export` beside the existing predictor. Do not add a flavor registry, export dataclass, model-registry wrapper, serving abstraction, or deployment configuration.
- `log_predictor` returns a plain mapping from declared form name to MLflow's native `ModelInfo`.
- The only stable form names are `pytorch` and `pyfunc`. The caller declares them explicitly; omission means no artifact for that form.

### Native MLflow contract

- Infer a single signature from numpy-compatible input and validated output fixtures, then pass it and the same input example to each native MLflow flavor logger.
- Use MLflow 3's immutable `ModelInfo.model_uri`, `model_id`, and `artifact_path`, plus the `LoggedModel.source_run_id`, as the recorded references. Do not create a DSIO registry or call MLflow Model Registry APIs.
- Attach checkpoint URI/digest and export-form identity as primitive MLflow model metadata/tags rather than duplicating them in a new DSIO object.

### Export compatibility

- Native PyTorch uses MLflow's pickle representation because the Predictor owns Python semantic validation and a structured identity-bearing mapping that cannot be represented by the PT2 tensor-only tracing contract.
- PyFunc converts MLflow tensor-signature arrays to the Predictor's tensor batch and converts native prediction tensors/sequences back to numpy arrays. It delegates all inference semantics to the same Predictor.
- Preflight every requested form and the representative fixture before logging. A logger/service failure may still leave native MLflow evidence of the attempted operation; DSIO does not invent transactions around MLflow.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 4.2, FR31, FR33, FR34]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Metrics, evaluation, and inference]
- [Source: `src/dsio/inference/predictor.py`]
- [Source: MLflow 3.16 `mlflow.pytorch.log_model`, `mlflow.pyfunc.log_model`, and Model Signature APIs]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 4.1 at `902e8a3`; verified MLflow 3.16 native flavor behavior locally and selected native `ModelInfo` references plus one private PyFunc adapter over any DSIO export hierarchy.
- 2026-09-22: Implemented native PyTorch/PyFunc export, one shared signature and input example, immutable source-Run linkage, flavor-attributed preflight, and actual serialize/load/execute equivalence checks.
- 2026-09-22: Adversarial review found and closed stale Run snapshots, wrong multi-form error attribution, forced CPU remapping, and lossy heterogeneous sequence conversion while preserving homogeneous NumPy scalar and NaN outputs.
- 2026-09-22: Exact candidate `dd67610` passed 19 focused tests, 910 core tests, 8 built-distribution/consumer tests, package build, Ruff, mypy, import contracts, and three independent final reviews.

### Completion Notes List

- `log_predictor` returns only MLflow `ModelInfo` values and logs exactly the explicitly requested native forms; DSIO adds no registry, export result, or deployment abstraction.
- Both representations package the complete semantic validator and are preflighted through deserialization and representative inference before the first MLflow model is initialized.
- Source Run state is revalidated across the operation, and exported model metadata/tags retain checkpoint and form provenance.

### File List

- `_bmad-output/implementation-artifacts/4-2-log-a-signed-mlflow-model-with-explicit-export-forms.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `src/dsio/inference/__init__.py`
- `src/dsio/inference/export.py`
- `tests/inference/test_export.py`

### Change Log

- 2026-09-22: Created Story 4.2 and started implementation.
- 2026-09-22: Completed native signed MLflow export with explicit forms and closed the independent acceptance, blind, and edge-case review gates.
