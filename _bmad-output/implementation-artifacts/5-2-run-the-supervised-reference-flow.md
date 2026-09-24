---
baseline_commit: ebf9ea6
---

# Story 5.2: Run the Supervised Reference Flow

Status: done

> Historical implementation record. ADR-0020 supersedes the parent/child tracking hierarchy used by this original acceptance run.

## Story

As a new DSio consumer,
I want an executable supervised example using only public APIs,
so that I can understand and verify the complete supported experiment spine.

## Acceptance Criteria

1. In a clean supported environment with local MLflow tracking, an ordinary project-owned Prefect flow builds synthetic data, generates and logs a split, trains through the exact `DsioModule` and `DsioDataModule`, evaluates, exports a predictor, and performs inference; every node has correctly linked native MLflow evidence.
2. Repeating the same declared inputs and seed in CI reproduces split assignments, execution identities, deterministic outputs, and validated lineage without private data, a consumer repository, a remote scheduler, or deployment infrastructure.
3. Given successful model and dataset evidence, a downstream-only flow with changed evaluation configuration creates new evaluation evidence without retraining and records immutable lineage to the reused sources.
4. The workflow is visible ordinary Prefect code outside DSio; it needs no hidden DSio DAG, CLI, runner, result model, or promotion gate.

## Tasks / Subtasks

- [x] Add one repository example that behaves like consumer-owned code (AC: 1, 4)
  - [x] Define native Prefect tasks and flows directly; keep orchestration outside `src/dsio`.
  - [x] Use only public DSio imports and native MLflow/Lightning objects.
- [x] Execute the complete synthetic supervised spine (AC: 1-2)
  - [x] Build canonical synthetic storage and immutable dataset evidence.
  - [x] Generate, validate, log, and replay one DSio split manifest.
  - [x] Train the exact DSio Lightning module/data module and log linked training evidence.
  - [x] Evaluate, export an immutable predictor, and perform signature-valid inference.
- [x] Demonstrate deterministic replay and selective downstream work (AC: 2-3)
  - [x] Repeat the full flow and compare identities, assignments, outputs, and lineage.
  - [x] Reevaluate from immutable source evidence with changed evaluation configuration and prove no retraining.
- [x] Cover the example through installed-distribution and CI gates (AC: 1-4)

## Dev Notes

### Minimal shape

- The example is project code, not a new DSio framework surface. Prefer one readable module; promote it to a package only when distinct responsibilities make the file too large.
- Pass MLflow Run IDs and immutable `runs:/...` URIs between ordinary Prefect tasks. Do not introduce a flow wrapper, DAG model, result dataclass, service layer, CLI, cache planner, or promotion concept.
- Reuse the existing synthetic store, split, tracking, Lightning, evaluation, export, and predictor contracts. Add a DSio abstraction only if the executable flow exposes a genuine missing public contract.
- Keep the CPU example tiny and deterministic enough for hosted CI.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 5.2, FR37]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — First generic acceptance slice]
- [Source: `tests/test_project_flow.py`]
- [Source: `README.md`]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 5.1 at `ebf9ea6`; chose an executable consumer-owned example over any new orchestration abstraction.
- 2026-09-22: Focused replay proof passed with stable split assignments, execution identities, predictions, and downstream-only evaluation.
- 2026-09-22: Full gate passed: 1046 tests, Ruff, mypy, import contracts, and wheel/sdist build.
- 2026-09-22: Review found replay-path and downstream-lineage gaps; added run-scoped stores, evidence/content validation, complete identities, and installed-wheel execution coverage.

### Completion Notes List

- Added an ordinary Prefect reference project outside the DSio distribution; it composes public DSio APIs and native Lightning/MLflow objects without a wrapper, runner, or result model.
- The full supervised flow records parent/child run lineage, split reuse, checkpoint-backed export, native evaluation/inference inputs, and output artifacts.
- The deterministic test executes two complete flows and a changed-metric evaluation-only flow while forbidding retraining.
- Exact-argument replay now uses isolated physical stores while preserving stable logical identities; downstream work fails closed for swapped stores, deleted Runs, or mismatched model/checkpoint evidence.
- Every Prefect task opens its MLflow attempt before fallible input loading, and training identity includes the dataset factory plus all declared trainer inputs.
- The built-distribution test copies the consumer project into an isolated environment and runs it against the installed wheel.
- Split the example tasks by data, training, and downstream responsibility once the initial module exceeded the agreed size boundary.

### File List

- `_bmad-output/implementation-artifacts/5-2-run-the-supervised-reference-flow.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `reference_projects/__init__.py`
- `reference_projects/supervised/__init__.py`
- `reference_projects/supervised/components.py`
- `reference_projects/supervised/flow.py`
- `reference_projects/supervised/tasks/__init__.py`
- `reference_projects/supervised/tasks/data.py`
- `reference_projects/supervised/tasks/downstream.py`
- `reference_projects/supervised/tasks/training.py`
- `tests/reference_flows/test_supervised_flow.py`

### Change Log

- 2026-09-22: Created Story 5.2 and started implementation.
- 2026-09-22: Implemented and validated the supervised reference flow; moved to review.
- 2026-09-22: Completed review gates and merged PR #54; marked done.
