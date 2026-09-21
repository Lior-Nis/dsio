---
baseline_commit: ed718aaabaafc52bc03562c79833b2173b82e332
---

# Story 1.4: Record Deterministic Identity and Safe Configuration

Status: done

## Story

As an experiment author,
I want each execution identified from its meaningful inputs and accompanied by a safe normalized configuration,
so that I can compare, reproduce, and reuse work without exposing credentials.

## Acceptance Criteria

1. **Given** semantically equivalent supported inputs with different mapping order or list/tuple and set/frozenset representations, **when** DSio normalizes and hashes them, **then** it produces the same execution identity independently of process or worker scheduling.
2. **Given** a relevant configuration, code, data, split, component, seed, environment, or DSio-version input changes, **when** identity is recalculated, **then** the execution identity changes.
3. **Given** recursive mapping fields declared secret or ephemeral change, **when** identity is recalculated, **then** the identity does not change, **and** the omitted field names and values are absent from the normalized representation.
4. **Given** a valid active native MLflow Run, **when** provenance is recorded, **then** its execution identity and normalized configuration are stored as native MLflow tags and one JSON artifact, **and** the artifact identifies the DSio version and explicit component references.
5. **Given** configuration contains declared credentials, **when** DSio normalizes, hashes, logs, or reports it, **then** secret field values are removed before canonicalization and never appear in MLflow parameters, tags, artifacts, or error messages.
6. **Given** a required input cannot be normalized deterministically, **when** identity is requested, **then** DSio raises `NonCanonicalValueError` identifying only the unsupported type or structural location, **and** never substitutes `str()`, `repr()`, or a memory-address representation.

## Tasks / Subtasks

- [x] Define the pure safe-normalization contract test-first (AC: 1, 3, 5, 6)
  - [x] Add `normalize(config, *, secrets=(), ephemeral=())` under `dsio.tracking.provenance`.
  - [x] Reuse `dsio.contracts.canonical_json` as the one canonical encoding rather than introducing a second serializer.
  - [x] Remove matching mapping fields recursively before their values are inspected; accept only explicit supported primitive/container representations.
- [x] Define execution identity as one plain digest (AC: 1-3, 6)
  - [x] Add `execution_identity(config, *, components=(), secrets=(), ephemeral=()) -> str`.
  - [x] Hash one normalized document containing schema version, installed DSio version, safe configuration, and explicit named component references.
  - [x] Prove every meaningful input changes the digest while secret and ephemeral changes do not.
- [x] Record provenance through native MLflow evidence (AC: 4, 5)
  - [x] Add `record_provenance(run_id, config, *, components=(), secrets=(), ephemeral=()) -> str`.
  - [x] Require an active, non-deleted native Run, then log `provenance.json` and searchable `dsio.execution_identity` / `dsio.version` tags with `MlflowClient`.
  - [x] Fail closed with `TrackingError` on Run resolution or evidence persistence without including configuration or secret values in errors.
  - [x] Return only the digest string; add no provenance/config/result model or repository abstraction.
- [x] Exercise the child-attempt integration and installed distribution (AC: 2, 4, 5)
  - [x] Record provenance inside `attempt(parent_run_id)` using the yielded native child ID.
  - [x] Assert artifact, tags, version, components, and safe normalized configuration through native MLflow reads.
  - [x] Extend the installed-wheel flow and README without adding project-specific configuration types.
- [x] Run the complete verification gate (AC: 1-6)
  - [x] Focused normalization, identity, MLflow, consumer-flow, distribution, and import-safety tests.
  - [x] Full pytest, Ruff, mypy, and import-linter suite.
  - [x] Lock validation, package build, and diff checks.

### Review Findings

- [x] [Review][Patch] Remove secret and ephemeral fields from lazy mappings before fetching their values. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Treat a bare string selector as one field name instead of a collection of characters. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Reject malformed non-string secret and ephemeral selectors instead of silently failing open. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Materialize one-shot field selectors once so validation cannot consume them before filtering. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Reject non-string component names with a safe `NonCanonicalValueError`. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Reject cyclic containers without exposing their values. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Preserve the semantic distinction between ordered sequences and unordered sets in the canonical identity. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Protect a Run's first recorded identity from later sequential replacement with an immutable native MLflow parameter. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Patch] Revalidate lifecycle after persistence so a Run finalized during the write fails closed. [`src/dsio/tracking/provenance.py`]
- [x] [Review][Disposition] Keep concurrent same-Run provenance writes outside this single-writer contract; MLflow FileStore offers no portable atomic compare-and-set, and an in-process lock would provide a false guarantee. [`src/dsio/tracking/provenance.py`]

## Dev Notes

### Minimal Public Contract

```python
from dsio.tracking import attempt, record_provenance

with attempt(parent_run_id) as child:
    identity = record_provenance(
        child.info.run_id,
        config,
        components={"dataset": "project.data:load_algae"},
        secrets={"password", "token"},
        ephemeral={"task_run_id"},
    )
```

- `normalize` returns a plain JSON-safe dictionary, `execution_identity` returns one SHA-256 string, and `record_provenance` returns that same string. These are values, not new domain entities.
- Secret and ephemeral selectors are field names matched at every mapping depth. To omit a whole credential structure, declare its containing field. This intentionally avoids a path-expression language.
- Project configuration remains project-owned. DSio neither validates its domain semantics nor introduces a base settings class.
- Component references are explicit `name -> import path/version reference` strings. Importability/admission is enforced in later component stories, not here.

### Normalization and Identity Requirements

- Filter declared fields before canonicalization so a secret value is never serialized, hashed, formatted into an error, or passed to MLflow.
- Reuse the existing canonical contract: string-keyed mappings, `None`, booleans, strings, integers, finite floats, lists/tuples, and sets/frozensets. Reject every other value by type.
- Mapping order is irrelevant; list and tuple are equivalent; set and frozenset are equivalent and order-independent. Sequence order remains meaningful.
- Hash a document with fixed keys: `schema_version`, `dsio_version`, `configuration`, and `components`. The installed distribution version is always identity-bearing.
- Do not infer repository, data, split, seed, hardware, or component inputs. Callers place every output-relevant value in configuration or components; hidden inference would make node identity unclear.

### Native MLflow Requirements

- Use `MlflowClient.get_run`, `log_dict`, and `set_tag` with the explicit child Run ID. Never activate or discover an ambient Run.
- Record one `provenance.json` artifact containing the normalized identity document plus its digest. Store the digest and DSio version as MLflow tags for Story 1.5 queries.
- Validate Run lifecycle before logging. Any partial native evidence remains visible on the failed attempt; never create a DSio transaction/result layer to hide it.
- Error messages name the operation and Run ID, never the configuration or its values.

### Architecture Compliance

- Keep provenance under `dsio.tracking`; MLflow remains the evidence store and project code remains the owner of typed configuration.
- Add no `ExecutionIdentity`, `Provenance`, `NormalizedConfig`, or result dataclass; no secret vault, environment scanner, logging proxy, or global context.
- Existing `dsio.contracts` remains the canonical serialization leaf. Existing legacy `dsio.runs.provenance` is not expanded or imported into the new spine.
- Evidence reuse belongs to Story 1.5; Prefect cache keys and downstream rerun behavior belong to Story 1.6.
- Keep root `import dsio` inert.

### Testing Requirements

- Prove equivalent containers and mapping order across spawned processes or hash seeds.
- Use sentinel secrets in unsupported objects and forced MLflow errors; search all native Run params, tags, and downloaded artifact bytes for absence.
- Mock only persistence failures; use the real isolated MLflow store for tags and artifacts.
- Exercise the built wheel outside the checkout.

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story-14-Record-Deterministic-Identity-and-Safe-Configuration]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Execution-identity-and-provenance]
- [Source: docs/superpowers/specs/2026-09-18-generic-experiment-spine.md#Package-shape]
- [Source: src/dsio/contracts/hashing.py]
- [Source: _bmad-output/implementation-artifacts/1-3-track-task-attempts-as-immutable-child-runs.md]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created from merged Story 1.3 at `ed718aa`; selected one safe normalized document shared by hashing and MLflow artifact logging.
- 2026-09-21: Implemented safe normalization and identity test-first, then added explicit native MLflow persistence and consumer-flow coverage.
- 2026-09-21: Complete pre-review gate passed with 625 tests, Ruff, mypy, four import contracts, lock validation, build, and diff checks.
- 2026-09-21: Adversarial review found secret-access, canonicalization, validation, immutability, and lifecycle gaps; all in-scope findings were patched test-first.
- 2026-09-21: Final blind, edge-case, and acceptance passes approved the complete patched diff with no remaining actionable issue.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Story is implementation-ready with three plain functions and no duplicate configuration, provenance, or result model.
- Added `normalize`, `execution_identity`, and `record_provenance` as three plain functions over one canonical document.
- Declared secret and ephemeral fields are recursively removed before unsupported values can be inspected, serialized, hashed, logged, or included in errors.
- Execution identity includes the installed DSio version and explicit component references and remains stable across mapping order, equivalent containers, processes, and hash seeds.
- Provenance is one native MLflow JSON artifact, one immutable identity parameter, and two searchable tags on the explicit Run ID; no ambient Run or DSio persistence model is used.
- Pre-review gate: 625 tests passed and 3 live tests were intentionally deselected; all static, architecture, lock, build, and diff checks passed.
- Review patches filter lazy mappings before access, preserve set semantics, validate one-shot selectors without consuming them, reject cyclic or structurally invalid input safely, protect the first recorded identity, and fail closed if lifecycle changes during persistence.
- Final post-review gate: 637 tests passed and 3 live tests were intentionally deselected; Ruff, mypy, four import contracts, lock validation, build, and diff checks passed.

### File List

- `_bmad-output/implementation-artifacts/1-4-record-deterministic-identity-and-safe-configuration.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/provenance.py`
- `tests/tracking/provenance/test_identity.py`
- `tests/tracking/provenance/test_mlflow.py`
- `tests/test_project_flow.py`
- `tests/test_built_distribution.py`

### Change Log

- 2026-09-21: Created Story 1.4 and marked it ready for development.
- 2026-09-21: Implemented Story 1.4 and moved it to review after the complete verification gate passed.
- 2026-09-21: Applied all in-scope adversarial review findings and documented concurrent same-Run writes as outside the single-writer provenance contract.
- 2026-09-21: Received final panel approval and marked Story 1.4 done after the post-review gate passed.
