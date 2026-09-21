---
baseline_commit: 91aaabdf65052f783f327c56edf9e4722e903968
---

# Story 1.5: Resolve and Reuse Verified MLflow Evidence

Status: done

## Story

As an experiment author,
I want a task to reuse previously produced evidence only when it is complete and provenance-compatible,
so that selective reruns remain fast without silently accepting stale or ambiguous results.

## Acceptance Criteria

1. Given an execution identity and required artifact paths, DSio searches native MLflow Runs and returns a native `Run` only after verifying successful status, active lifecycle, matching immutable identity evidence, provenance, and required artifacts.
2. Missing, failed, incomplete, deleted, incompatible, or inaccessible candidates are never returned as reusable; inability to query or establish validity fails closed with `TrackingError`.
3. Explicit evidence validation accepts only an immutable MLflow Run ID and produces `runs:/<run-id>/<artifact>` URIs; mutable aliases, stages, and malformed paths are rejected with the required immutable form identified.
4. A consuming attempt records the source Run ID and immutable evidence URI through ordinary safe provenance inputs; source evidence is referenced, not copied or modified.
5. Pure serializable Prefect tasks may use an identity-derived DSio cache-key function, while MLflow-producing tasks resolve evidence through MLflow and no parallel cache store is added.

## Tasks / Subtasks

- [x] Define resolution and validation contracts test-first (AC: 1-3)
  - [x] Add `resolve_evidence(identity, *, required_artifacts=()) -> Run | None`.
  - [x] Add `require_evidence(run_id, *, identity=None, required_artifacts=()) -> Run`.
  - [x] Validate native status, lifecycle, immutable identity parameter/tag, provenance artifact, and required artifact existence.
- [x] Keep references native and immutable (AC: 3-4)
  - [x] Add `evidence_uri(run_id, artifact_path) -> str` for exact `runs:/` artifact references.
  - [x] Reject aliases, stages, traversal, absolute paths, and malformed Run IDs.
  - [x] Demonstrate source Run ID and URI as ordinary inputs to `record_provenance` without a lineage model.
- [x] Use native Prefect caching only for pure values (AC: 5)
  - [x] Add one deterministic `prefect_cache_key(context, parameters)` function derived from `execution_identity`.
  - [x] Exercise it through Prefect's native `cache_key_fn`; add no DSio cache storage or task wrapper.
- [x] Preserve distribution and architecture boundaries (AC: 1-5)
  - [x] Re-export only the plain functions from `dsio.tracking`; keep root `import dsio` inert.
  - [x] Update the consumer flow, README, and installed-wheel proof.
- [x] Run the complete verification and review gate (AC: 1-5)
  - [x] Focused MLflow resolution, Prefect cache, consumer-flow, distribution, and import-safety tests.
  - [x] Full pytest, Ruff, mypy, import-linter, lock, build, and diff checks.
  - [x] Independent blind, edge-case, and acceptance review with all findings resolved or explicitly dispositioned.

### Review Findings

- [x] [Review][Patch] Include Prefect's stable task key so different pure tasks with identical parameters cannot share cached results. [`src/dsio/tracking/cache.py`]
- [x] [Review][Patch] Use Prefect's native stable task key without inspecting callable implementation types. [`src/dsio/tracking/cache.py`]
- [x] [Review][Patch] Reject current-directory, URI delimiter, and control-character artifact paths and prove accepted URIs round-trip through MLflow exactly. [`src/dsio/tracking/evidence/references.py`]
- [x] [Review][Patch] Require the exact schema-v1 provenance fields so unhashed additions cannot be accepted as reusable evidence. [`src/dsio/tracking/evidence/resolution.py`]
- [x] [Review][Patch] Wrap MLflow client-construction failures as actionable `TrackingError` values while preserving cancellation. [`src/dsio/tracking/evidence/resolution.py`]
- [x] [Review][Patch] Re-read lifecycle and identity metadata after artifact validation so evidence deleted or changed mid-validation is not returned. [`src/dsio/tracking/evidence/resolution.py`]
- [x] [Review][Patch] Treat non-UTF-8 or directory-shaped provenance as invalid candidates so they cannot shadow older valid evidence. [`src/dsio/tracking/evidence/resolution.py`]
- [x] [Review][Patch] Treat JSON parser limit/recursion failures as invalid candidates and enforce exact schema-v1 scalar/component types. [`src/dsio/tracking/evidence/resolution.py`]

## Dev Notes

### Minimal Public Contract

```python
from dsio.tracking import evidence_uri, prefect_cache_key, resolve_evidence

prior = resolve_evidence(identity, required_artifacts={"model/model.pkl"})
if prior is not None:
    model_uri = evidence_uri(prior.info.run_id, "model")
```

- Return MLflow's native `Run`, `None`, or a plain immutable URI string. Add no evidence, validation-result, cache, repository, or lineage class.
- A missing usable candidate is a normal cache miss (`None`). MLflow query/storage failures and explicit-reference validation failures raise `TrackingError`.
- The execution identity parameter is the immutable lookup key. The identity tag and `provenance.json` must agree with it before reuse.
- Select the newest verified candidate deterministically; invalid candidates never shadow an older valid candidate.
- `required_artifacts` contains exact relative POSIX artifact paths. Validation checks existence without copying source evidence into the consumer Run.
- Consuming projects record `source_run_id` and the exact `runs:/` URI as normal identity-bearing configuration passed to `record_provenance`.
- `prefect_cache_key` is only for pure serializable task parameters. Evidence-producing tasks call `resolve_evidence` instead.

### Architecture Compliance

- MLflow remains the evidence store and Prefect remains the cache/orchestration owner.
- Add no parallel cache, graph, result object, model registry, alias resolver, or project-specific policy.
- Keep evidence resolution under `dsio.tracking`; do not expand the legacy `dsio.runs` layer.
- Keep root `import dsio` inert.

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created from merged Story 1.4 at `91aaabd`; selected native Run/URI returns and a normal `None` cache miss to avoid a duplicate result model.
- 2026-09-21: Implemented resolution, explicit validation, immutable URI construction, and the pure Prefect cache key test-first.
- 2026-09-21: Complete pre-review gate passed with 659 tests, Ruff, mypy, four import contracts, lock validation, build, and diff checks.
- 2026-09-21: Adversarial review found cache identity, URI parsing, provenance schema/parser, client construction, and mid-validation lifecycle gaps; all were patched test-first.
- 2026-09-21: Final blind, edge-case, and acceptance passes unanimously approved the complete staged diff.

### Completion Notes List

- Added native MLflow Run resolution that validates lifecycle, status, parameter/tag agreement, recomputed provenance identity, DSio version, and exact required artifacts.
- Added explicit Run validation and immutable `runs:/` URI construction without an evidence wrapper or alias layer.
- Added one Prefect-native cache-key function for pure serializable task values; MLflow evidence continues to resolve only through MLflow.
- Demonstrated reference-only reuse in project-owned flows and the installed distribution without copying or modifying source evidence.
- Pre-review gate: 659 tests passed and 3 live tests were intentionally deselected; all static, architecture, lock, build, and diff checks passed.
- Review patches bind cache keys to Prefect task identity, enforce exact MLflow URI paths and provenance schema, skip malformed candidates without hiding store outages, and revalidate the final native Run snapshot.
- Final post-review gate: 674 tests passed and 3 live tests were intentionally deselected; Ruff, mypy, four import contracts, lock validation, build, and diff checks passed.

### File List

- `_bmad-output/implementation-artifacts/1-5-resolve-and-reuse-verified-mlflow-evidence.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/cache.py`
- `src/dsio/tracking/evidence/__init__.py`
- `src/dsio/tracking/evidence/references.py`
- `src/dsio/tracking/evidence/resolution.py`
- `tests/tracking/evidence/test_references.py`
- `tests/tracking/evidence/test_resolution.py`
- `tests/tracking/test_cache.py`
- `tests/test_project_flow.py`
- `tests/test_built_distribution.py`

### Change Log

- 2026-09-21: Created Story 1.5 and started test-first implementation.
- 2026-09-21: Implemented Story 1.5 and moved it to review after the complete verification gate passed.
- 2026-09-21: Applied all adversarial review findings and received unanimous final approval.
- 2026-09-21: Marked Story 1.5 done after the exact post-review gate passed.
