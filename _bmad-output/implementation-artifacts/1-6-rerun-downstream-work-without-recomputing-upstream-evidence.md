---
baseline_commit: 1dd9aefa3f05e9f3cd3834427e6ee9105d3abb0b
---

# Story 1.6: Rerun Downstream Work Without Recomputing Upstream Evidence

Status: done

## Story

As an experiment author,
I want a new flow execution to consume immutable evidence from an earlier execution,
so that I can change and rerun downstream analysis without repeating valid upstream work.

## Acceptance Criteria

1. A project-owned flow validates exact successful model/data Run evidence before invoking downstream computation; DSio invokes no upstream task.
2. Every downstream-only execution creates a fresh parent and child Run and records source Run IDs and immutable artifact URIs as identity-bearing provenance.
3. Changing downstream configuration changes only the downstream identity and evidence; source artifacts remain unchanged.
4. Task selection and ordering remain ordinary project Python/Prefect logic; DSio adds no planner, DAG model, scheduler, or rerun abstraction.
5. Mutable, missing, failed, or incompatible source evidence fails before downstream computation with an actionable contract error.
6. Successful downstream output is logged beneath the new parent while the original source remains unchanged and auditable.

## Tasks / Subtasks

- [x] Prove the selective-rerun vertical test-first (AC: 1-6)
  - [x] Produce one successful source Run with model and dataset artifacts.
  - [x] Define a downstream-only project flow that validates source evidence before submitting its task.
  - [x] Run two downstream configurations and assert fresh parents/children, distinct downstream identities, shared immutable source URIs, and unchanged source bytes.
  - [x] Prove invalid source evidence prevents downstream computation.
- [x] Keep the architecture minimal (AC: 1, 4)
  - [x] Reuse `experiment`, `attempt`, `require_evidence`, `evidence_uri`, and `record_provenance` without adding a public API.
  - [x] Keep selection, ordering, and validation calls visible in project-owned Prefect code.
  - [x] Update README and installed-consumer documentation only where the selective-rerun pattern needs clarification.
- [x] Run complete verification and review gates (AC: 1-6)
  - [x] Focused downstream-rerun, tracking, consumer-flow, and distribution tests.
  - [x] Full pytest, Ruff, mypy, import-linter, lock, build, and diff checks.
  - [x] Independent blind, edge-case, and acceptance review.

### Review Findings

- [x] [Review][Patch] Open the fresh parent before source validation so invalid evidence records a `FAILED` parent while still invoking no downstream task or child Run. [`tests/tracking/test_downstream_rerun.py`]
- [x] [Review][Patch] Prove identical downstream inputs still create fresh Runs, compare every source metadata/artifact byte, read result payloads, and exercise failed/missing/incompatible/mutable sources. [`tests/tracking/test_downstream_rerun.py`]
- [x] [Review][Disposition] Retain a failed parent for every invalid downstream flow execution; the generic-spine lifecycle requires failed validation to remain visible in MLflow, so “no parent Run” is not accepted. [`tests/tracking/test_downstream_rerun.py`]

## Dev Notes

- This is an executable integration story over the Story 1.2–1.5 public API. Add no rerun helper, planner, source bundle, evidence model, or lineage class unless the existing primitives genuinely cannot satisfy an acceptance criterion.
- Validate exact source Run identity and required artifacts in the project flow before calling the downstream task.
- Pass source Run IDs and `runs:/` URIs as ordinary task inputs and record them in downstream provenance.
- A configuration-only downstream change must not touch source Run metadata or artifact bytes.
- Keep root `import dsio` inert.

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created from merged Story 1.5 at `1dd9aef`; selected an integration-only implementation because existing native primitives satisfy the story.
- 2026-09-21: Complete pre-review gate passed with 675 tests, Ruff, mypy, four import contracts, lock validation, build, and diff checks.
- 2026-09-21: Final blind, edge-case, and acceptance passes unanimously approved the strengthened integration proof.

### Completion Notes List

- Proved downstream-only reruns through ordinary project-owned Prefect code without adding a DSio rerun API.
- Source evidence is validated before task submission, referenced by exact native URIs, recorded in child provenance, and remains byte-for-byte unchanged.
- Changed downstream configuration creates fresh parent/child evidence and a distinct identity; failed source validation invokes no downstream computation.
- Pre-review gate: 675 tests passed and 3 live tests were intentionally deselected; all static, architecture, lock, build, and diff checks passed.
- Review patches keep invalid validation visible as failed parents, prove identical-input executions remain fresh, cover every invalid source class, compare complete source metadata/artifacts, and validate downstream result bytes.
- Final post-review gate: 675 tests passed and 3 live tests were intentionally deselected; Ruff, mypy, four import contracts, lock validation, build, and diff checks passed.

### File List

- `_bmad-output/implementation-artifacts/1-6-rerun-downstream-work-without-recomputing-upstream-evidence.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `tests/tracking/test_downstream_rerun.py`

### Change Log

- 2026-09-21: Created Story 1.6 and started the selective-rerun integration proof.
- 2026-09-21: Implemented the integration-only Story 1.6 delta and moved it to review after the complete gate passed.
- 2026-09-21: Applied review-strengthening findings, received unanimous approval, and marked Story 1.6 done.
