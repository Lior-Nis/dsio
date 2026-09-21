---
baseline_commit: 487bd79b59a3e0bd4960bcd57c0cf520e699f061
---

# Story 2.2: Read and Write Samples Through One Store Interface

Status: review

## Story

As a dataset author,
I want to build and reopen a DSio store through one stable interface,
so that training code is independent of the selected physical format.

## Acceptance Criteria

1. A completed store returns every persisted sample by stable string identity or deterministic zero-based position through one public method.
2. Each returned sample preserves the supported fields: `sample_id`, numeric `data`, leakage `group`, and JSON-compatible `attrs`; data shape and dtype match what was built, and the manifest records the store schema version.
3. Reopened stores and spawned worker processes lazily memory-map the payload per process; concurrent reads are deterministic and the store object does not serialize payload bytes.
4. Opening rejects missing, truncated, malformed, topologically inconsistent, or unsupported-schema metadata with `StoreError` naming the file/store invariant; reading a sample rejects content whose sample digest changed.
5. The implementation remains one flat-binary store with no backend argument, registry, format branch, duplicate store class, or second persistence model.
6. Existing row/window consumers and public imports remain compatible while the oversized `store.py` becomes a cohesive `store/` package.

## Tasks / Subtasks

- [x] Define the narrow stable sample contract test-first (AC: 1-2)
  - [x] Treat each persisted entity/recording as one storage-boundary sample; derived window identities remain the responsibility of views/datasets.
  - [x] Add `read_sample(sample_id_or_position)` returning a typed mapping with the four supported fields.
  - [x] Make `read_entity()` delegate to the same checked data path and preserve existing row/window methods.
  - [x] Persist a schema version and per-sample data digest without adding a parallel schema/result object.
- [x] Fail closed at the store boundary (AC: 2, 4)
  - [x] Validate manifest/index/entity agreement, payload byte size, entity ordering/offsets/counts, unique identities, group count, and supported schema during open.
  - [x] Wrap malformed JSON/YAML/index/filesystem errors as actionable `StoreError` messages.
  - [x] Validate the selected sample's content digest on identity- and position-based reads.
- [x] Preserve memory-mapped concurrency (AC: 3)
  - [x] Keep readers lazy, per PID, and absent from pickle state.
  - [x] Prove deterministic reads from spawned worker processes without loading the full payload into the serialized store object.
- [x] Promote the large module into a package (AC: 5-6)
  - [x] Split on-disk schema/layout, building, and reading/validation into cohesive modules under `dsio.data.store`.
  - [x] Re-export the existing public names from `dsio.data.store`; remove the unused `Window` result class rather than carrying dead API.
  - [x] Add no generic backend protocol, registry, alternate store class, or runtime dependency.
- [x] Verify and review (AC: 1-6)
  - [x] Run focused store, dataset, view, split, training, and distribution tests.
  - [x] Run full pytest, Ruff, mypy, import-linter, lock validation, build, and diff checks.
  - [ ] Complete independent blind, edge-case, and acceptance reviews before merge.

## Dev Notes

### Minimal contract

- Deepen `SignalStore`; do not introduce `Store`, `SampleStore`, an ABC, or a second physical representation. The existing name remains compatible and its flat-binary layout already serves signals, token sequences, and flattened fixed-size items.
- At this boundary, a persisted `Entity.entity_id` is the stable storage sample identity. A future window dataset derives its own training `sample_id` from store identity plus window provenance; do not conflate or prebuild that Story 2.5 concern.
- `read_sample(key)` accepts only `str` identity or non-negative `int` position. It returns a `TypedDict`, not a dataclass/Pydantic result model: `sample_id`, `data`, `group`, `attrs`.
- The store supports one homogeneous numeric `data` field selected by Story 2.1. `sample_id`, `group`, and `attrs` are metadata fields. Do not add arbitrary tensor-field plumbing before a reference flow requires it.

### Integrity boundary

- Preserve the existing whole-payload/index/entity digests and explicit `verify()` method.
- Add a digest to each entity record so `read_sample`/`read_entity` can detect sample-local payload corruption without hashing a multi-gigabyte store on every open.
- Opening should perform bounded structural validation: required files, parseability, schema version, payload size, manifest/header agreement, small-file digests, entity/offset topology, identity uniqueness, and group counts.
- Do not hash the whole signal payload during every open. `verify()` remains the deliberate full integrity scan; sample reads verify only the selected entity.
- Partial row reads remain the low-level window path and preserve their current boundary checks. They are not the stable whole-sample API introduced here.

### Package shape

- Promote `src/dsio/data/store.py` to `src/dsio/data/store/` because the current 397-line file now has distinct on-disk schema, builder, and reader/validation responsibilities.
- Suggested cohesive files: `layout.py`, `builder.py`, `reader.py`, and `__init__.py`. Preserve `from dsio.data.store import ...` through re-exports.
- Keep `src/dsio/data/readers.py` as the small physical mmap primitive selected in Story 2.1.

### Regression guardrails

- Moving a store directory must remain valid; manifest `name` is descriptive and must not bind integrity to an absolute path.
- A builder exception may leave an unpublished directory, but opening it must fail clearly and never return partial data.
- Reject negative positions, booleans masquerading as integers, unknown identities, duplicate normalized identities, corrupt sample bytes, and metadata that disagrees with the binary index.
- Root `import dsio` remains inert. No import-time I/O or optional benchmark dependency enters production.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Epic 2, Story 2.2, FR21]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Data and splits; Package shape]
- [Source: `_bmad-output/implementation-artifacts/2-1-select-the-primary-store-through-a-reproducible-benchmark.md`]
- [Source: `docs/adr/0005-canonical-store-is-flat-binary.md`]
- [Source: `src/dsio/data/store.py`]
- [Source: `src/dsio/data/format.py`]
- [NumPy memory mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 2.1 at `487bd79`; inspected Epic 2, the accepted spine, the complete current store and format modules, store consumers, tests, and recent history.
- 2026-09-22: Wrote the public store-interface tests first and confirmed 12 expected failures before implementation.
- 2026-09-22: Focused store tests: 73 passed. Broader data/dataset/eval/model/train/distribution tests: 466 passed, 3 deselected.
- 2026-09-22: Full gate: 694 passed, 3 deselected; Ruff, mypy (64 source files), four import contracts, lock validation, build, and diff checks passed.
- 2026-09-22: Review found canonical-offset, constructor-validation, strict-schema, JSON-attrs, and builder-cleanup gaps. Added failing regressions, applied minimal boundary fixes, and reran the full gate: 716 passed, 3 deselected; all other gates remained green.

### Completion Notes List

- Deepened the one existing `SignalStore` with identity- and position-based whole-sample reads; no backend selector or duplicate store abstraction was introduced.
- Added a versioned layout and per-sample digests, with bounded validation on open and explicit full verification retained.
- Preserved lazy per-process memory mapping and proved deterministic spawn-worker reads without serializing payload bytes.
- Split the former 397-line module into cohesive layout, builder, and reader modules while preserving public imports and row/window consumers.
- Review hardening now rejects impossible index topology, invalid channels/dtypes, coercive schema versions, and non-JSON attributes before partial data can escape; legacy unversioned stores fail with an explicit rebuild instruction.

### File List

- `_bmad-output/implementation-artifacts/2-2-read-and-write-samples-through-one-store-interface.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/data/store.py` (removed)
- `src/dsio/data/store/__init__.py`
- `src/dsio/data/store/builder.py`
- `src/dsio/data/store/layout.py`
- `src/dsio/data/store/reader.py`
- `tests/data/test_store_interface.py`

### Change Log

- 2026-09-22: Created Story 2.2 and started implementation.
- 2026-09-22: Implemented and verified the stable, content-checked store interface; moved to review.
- 2026-09-22: Addressed all first-pass blind, edge-case, and acceptance findings and requested closure review.
