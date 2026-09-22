---
baseline_commit: b5da1696
---

# Story 2.4: Record and Reload Split Evidence in MLflow

Status: in-progress

## Story

As an experiment author,
I want split data and its manifest recorded with native MLflow lineage,
so that later training and evaluation Runs can reuse the exact dataset evidence.

## Acceptance Criteria

1. Recording a validated `Examples` and `SplitFile` on a running tracked task logs the source as a native MLflow dataset input and stores the full split YAML at a content-addressed artifact path containing its verified digest and dataset identity.
2. Reloading from an exact immutable `runs:/<run-id>/<artifact-path>` reference requires a finished, active, provenance-valid MLflow Run; it downloads and verifies the artifact digest, source dataset identity, and all concrete split invariants before returning the existing `SplitFile`.
3. Missing artifacts or dataset lineage, mutable/failed/deleted Runs, corrupt manifests, identity mismatches, and invalid assignments fail closed with an error that names the failed evidence check before any DataLoader construction.
4. Successful reuse logs the source Run's same native MLflow dataset entity as an input to the running consumer Run, tagged with the immutable `runs:/...` manifest URI; it does not copy the source artifact or persist a DSIO dataset/result/reference model.
5. Recording and loading use explicit Run IDs, create no implicit active Run, preserve cancellation, and leave root `import dsio` inert.

## Tasks / Subtasks

- [x] Add the minimal split-evidence API under tracking (AC: 1-5)
  - [x] Expose `record_split_evidence()` and `load_split_evidence()` from `dsio.tracking.evidence` and `dsio.tracking`.
  - [x] Return/use native immutable Run/artifact strings and the existing `SplitFile`; add no result or reference dataclass.
  - [x] Reuse one shared running-Run guard for provenance and split evidence.
- [x] Record native MLflow evidence (AC: 1, 5)
  - [x] Validate the concrete examples and manifest before any MLflow write.
  - [x] Log one native `MetaDataset`/`DatasetInput` with source, dataset digest, split context, manifest digest, and immutable artifact URI.
  - [x] Save the full YAML under `split-evidence/<digest>/manifest.yaml` and return its `runs:/` URI.
- [x] Reload and link immutable evidence (AC: 2-4)
  - [x] Require the source Run through existing provenance-aware evidence validation and require the consumer Run to be writable.
  - [x] Verify the referenced path is a file, parse the YAML, verify its persisted digest and content-addressed path, and validate it against the supplied examples.
  - [x] Require the matching native source dataset input, then log that same MLflow dataset entity on the consumer with source-artifact lineage tags.
  - [x] Perform every validation before writing consumer lineage or returning the manifest.
- [ ] Verify and review (AC: 1-5)
  - [x] Cover happy-path round-trip and exact native MLflow dataset/artifact lineage with the real local MLflow backend.
  - [x] Cover running/failed/deleted sources, missing/corrupt/mismatched artifacts, absent or mismatched dataset inputs, invalid consumers, and MLflow failures.
  - [x] Run focused and full tests, Ruff, mypy, import contracts, lock validation, build, and diff checks.
  - [ ] Complete independent blind, edge-case, and acceptance reviews before merge.

## Dev Notes

### Minimal public shape

- MLflow owns the Run, dataset entity, dataset-input association, artifact, and lineage tags. DSIO contributes only split-specific validation and orchestration.
- `record_split_evidence(run_id, examples, manifest, source=...) -> str` returns the immutable `runs:/` artifact URI.
- `load_split_evidence(manifest_uri, examples, *, consumer_run_id) -> SplitFile` directly consumes the URI returned by recording and returns the already-established manifest model.
- Do not add `SplitEvidence`, `DatasetRef`, `ArtifactRef`, a repository, or another persistence hierarchy.

### Native MLflow mapping

- Represent source metadata with MLflow's native `MetaDataset` and caller-supplied native `DatasetSource`; paths/strings normalize to `LocalArtifactDatasetSource` for the primary memmap store.
- Use `MlflowClient.log_inputs()` with native `DatasetInput`/`InputTag` entities because tracked attempts are explicit client-created Runs, not implicit fluent active Runs.
- Source inputs use context `split`; consuming inputs use context `split_reuse` and point to the immutable manifest URI. Reuse the exact source `Dataset` entity rather than reconstructing or copying it.
- A failed partial write remains on a non-finished task Run and is therefore not reusable.

### Validation order

- Recording validates `Examples` consistency and `validate(examples, manifest)` before resolving the writable Run or writing artifacts.
- Loading validates both Run references, then the source Run/provenance/artifact requirement, YAML/digest/path, concrete dataset invariants, and native source dataset input. Only after all checks pass may it log the consumer input.
- `SplitFile.load()` remains the digest/schema boundary; `dsio.data.splits.validate()` remains the concrete identity/leakage/coverage boundary.
- MLflow availability failures are not cache misses here: wrap ordinary failures as actionable `TrackingError` and preserve cancellation/base exceptions.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 2.4, FR26]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Data and splits, MLflow owns evidence]
- [Source: `src/dsio/tracking/evidence/resolution.py`]
- [Source: `src/dsio/tracking/evidence/references.py`]
- [Source: `src/dsio/data/splits/models/`]
- [Source: `src/dsio/data/splits/validation.py`]
- [MLflow 3.16.1 native `MetaDataset`, `DatasetInput`, `InputTag`, and `MlflowClient.log_inputs()` APIs]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 2.3 at `b5da1696`; inspected existing immutable evidence resolution/provenance, native MLflow 3.16.1 dataset APIs, and split validation boundaries.
- 2026-09-22: Implemented direct `runs:/` record/load composition with native `MetaDataset`/`DatasetInput` lineage, content-addressed manifest artifacts, provenance-aware source reuse, and concrete split validation.
- 2026-09-22: Local gate passed: 768 tests passed (3 deselected), Ruff, mypy, all three import contracts, lock validation, build, and diff checks.
- 2026-09-22: Independent review exposed MLflow's silent dataset-input deduplication, a source lifecycle race, ambiguous source lineage, and unnormalized source serialization failures; fixed each with exact persisted-input checks and refreshed source validation.
- 2026-09-22: Post-review gate passed: 774 tests passed (3 deselected), Ruff, mypy, all three import contracts, lock validation, build, and diff checks. The repository-wide Ruff format check remains outside CI and reports pre-existing formatting drift; both changed Python files pass formatting.
- 2026-09-22: Edge re-review found that a manifest URI could already belong to another native dataset identity; collision checks now cover both MLflow dataset identity and immutable manifest URI on source and consumer Runs.

### Completion Notes List

- The source Run owns one native MLflow dataset input and one content-addressed manifest; the consumer links the exact same native dataset entity and immutable manifest URI without copying artifacts.
- Reuse fails closed on mutable or invalid Runs, malformed references, missing/corrupt/wrongly-addressed manifests, source dataset mismatch, and partial MLflow writes.
- Record/reuse retries are idempotent, while conflicting lineage for MLflow's same native dataset identity is rejected before a silent deduplicated write and verified again from persisted Run state.
- The public API adds two functions and returns only a native URI or the existing `SplitFile`; no parallel evidence/result/reference model was introduced.

### File List

- `_bmad-output/implementation-artifacts/2-4-record-and-reload-split-evidence-in-mlflow.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `src/dsio/tracking/__init__.py`
- `src/dsio/tracking/_lifecycle.py`
- `src/dsio/tracking/evidence/__init__.py`
- `src/dsio/tracking/evidence/splits.py`
- `src/dsio/tracking/provenance.py`
- `tests/tracking/evidence/test_split_evidence.py`

### Change Log

- 2026-09-22: Created Story 2.4 and started implementation.
- 2026-09-22: Added native MLflow split evidence recording/reload and passed the complete local quality gate.
