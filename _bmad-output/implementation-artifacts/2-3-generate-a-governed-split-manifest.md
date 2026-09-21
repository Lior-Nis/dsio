---
baseline_commit: 8d946f9c
---

# Story 2.3: Generate a Governed Split Manifest

Status: in-progress

## Story

As a dataset author,
I want to generate named data roles through a DSIO-owned splitter,
so that assignments are deterministic, validated, reviewable, and replayable across projects.

## Acceptance Criteria

1. `dsio.data.splits.generate()` accepts an `Examples`, a known algorithm name, validated parameters, a seed, and role names, then returns the existing split document enriched with dataset identity, algorithm/version, normalized parameters, dependency provenance, validations, exact sample assignments, and a content digest.
2. Repeating generation with identical inputs produces byte-identical normalized assignments and the same digest; input ordering and worker completion ordering do not affect the result when dataset identity and membership are unchanged.
3. Established grouped algorithms delegate to the locked scikit-learn implementation while DSIO validates inputs, normalizes output, records the scikit-learn version, and proves role disjointness, group leakage safety, identity membership, and declared coverage.
4. The existing purged walk-forward implementation is reachable through the same dispatcher and records partial coverage caused by purge/embargo rather than silently assigning discarded samples.
5. Unknown identities, duplicate identities, role overlap, missing required roles, group leakage, and violated total/partial coverage fail with actionable `SplitError` messages.
6. The dispatcher is closed: unknown algorithm names and any runtime registration attempt fail with guidance to add a reviewed DSIO experimental algorithm; no registry or plugin hook is exposed.
7. Splits live under `dsio.data.splits`, use the one existing `SplitFile`/`SplitFold` contract rather than a parallel result hierarchy, and existing fold/training consumers continue to work through updated imports.

## Tasks / Subtasks

- [x] Put split ownership under the data domain (AC: 7)
  - [x] Move the cohesive split package to `dsio.data.splits` and update internal consumers and import contracts.
  - [x] Keep one manifest model by deepening `SplitFile`/`SplitFold`; do not add a duplicate evaluation or generation-result model.
- [x] Make stable example identity part of the split boundary (AC: 1, 5)
  - [x] Add aligned, unique `sample_ids` to the `Examples` protocol and shipped adapters.
  - [x] Derive deterministic window identities from entity identity plus row start; entity examples use persisted store identities.
  - [x] Preserve identities through subsetting and extend the reusable Examples contract tests.
- [x] Evolve the manifest without speculative abstractions (AC: 1-5)
  - [x] Record algorithm/version, normalized parameters, seed, dependency versions, validations, exact role assignments, and digest on the existing document.
  - [x] Validate role and assignment invariants on construction/load and verify the persisted digest.
  - [x] Resolve exact assignments when present while retaining group/temporal checks as leakage evidence.
- [x] Add one closed dispatcher (AC: 1-6)
  - [x] Support `group_kfold`, `stratified_group_kfold`, `group_shuffle`, `leave_one_group_out`, and existing `purged_walk_forward` through private branches.
  - [x] Use scikit-learn for established grouped algorithms and record its locked version; keep temporal logic native.
  - [x] Normalize all role/group/sample ordering and validate the finished manifest against its source examples.
  - [x] Reject unknown names with the governed experimental-admission instruction and expose no registration API.
- [ ] Verify and review (AC: 1-7)
  - [x] Add test-first determinism, replay, provenance, invariant, temporal, dispatcher-closure, and serialization coverage.
  - [x] Run focused split, examples, dataset, evaluation, training, and distribution tests.
  - [x] Run full pytest, Ruff, mypy, import-linter, lock validation, build, and diff checks.
  - [ ] Complete independent blind, edge-case, and acceptance reviews before merge.

## Dev Notes

### Minimal shape

- `SplitFile` already is the manifest. Add the missing evidence to it; do not introduce `SplitResult`, `GeneratedSplit`, or an algorithm class hierarchy.
- One public function dispatches through explicit internal branches. There is no mutable mapping, entry point, decorator, registration function, project callback, or fallback import path.
- Parameter validation stays close to each branch. Five small algorithm branches are clearer than a generic splitter protocol whose only caller is the dispatcher.
- Named roles remain strings. Default grouped roles are `train` and `test`, but callers may rename them; Lightning phase mapping remains Story 2.5.

### Identity and replay

- Exact assignment replay needs per-example stable identity. Add `sample_ids` to `Examples`; do not infer identity from Python object identity.
- `TableExamples` accepts explicit identities and defaults to deterministic string positions for existing callers. `SignalExamples` uses entity identity, row start, and view digest; entity-level examples use persisted `Entity.entity_id`.
- Normalize stored role assignments and group lists lexically. A manifest digest is over semantic content, not sklearn iterator order or dictionary insertion order.
- Group membership remains evidence and a leakage guard even when exact sample assignments drive replay.

### Algorithms and provenance

- `group_kfold`, `stratified_group_kfold`, `group_shuffle`, and `leave_one_group_out` delegate to scikit-learn 1.9's model-selection implementations. Scikit-learn becomes a runtime dependency because these are first-class DSIO paths, not optional project scripts.
- `purged_walk_forward` reuses `TemporalSpec`, `walk_forward`, and `apply`; no second temporal algorithm is created.
- Record DSIO algorithm version separately from dependency versions. A dependency upgrade changes manifest evidence even when assignments happen to remain equal.

### Validation

- Require unique source identities and aligned groups/attributes/times before generation.
- Every produced role is non-empty, exact assignments are mutually disjoint, required roles exist, assigned identities belong to the source, and grouped algorithms never divide one leakage group across roles within a fold.
- Total-coverage algorithms assign every source sample exactly once per fold. Purged temporal folds explicitly record partial coverage and the discarded count.
- Persisted digests are verified on load; changing assignments or provenance invalidates the document before resolution.

### Package migration

- The accepted 2026-09-18 spine and FR23 supersede ADR 0006's old “project-generated” boundary. Amend the ADR rather than retaining both designs.
- Move `dsio.splits` to `dsio.data.splits` as the canonical path and update repository consumers. Do not keep a second implementation package.
- Root `import dsio` remains inert and performs no generation or I/O.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 2.3, FR23-FR25]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Data and splits]
- [Source: `docs/adr/0006-splits-are-committed-group-lists.md`]
- [Source: `src/dsio/data/splits/models/`]
- [Source: `src/dsio/data/splits/resolve.py`]
- [Source: `src/dsio/data/splits/temporal.py`]
- [Source: `src/dsio/data/examples.py`]
- [scikit-learn grouped cross-validation API, locked locally at 1.9.0]

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-22: Created from merged Story 2.2 at `8d946f9`; reconciled the accepted generic spine and Epic 2 against the older project-generated-split ADR, inspected all split/example consumers, and verified the locked scikit-learn 1.9 grouped-split APIs.
- 2026-09-22: Implemented the closed dispatcher, stable example identity, schema-v3 exact-assignment manifests, digest/provenance verification, concrete-source validation, package migration, and compatibility updates.
- 2026-09-22: Self-audit tightened temporal-bound replay, cross-fold custom-role checks, count evidence, and malformed-YAML handling; promoted the oversized model module to a cohesive package.
- 2026-09-22: Local gate passed: 739 tests passed (3 deselected), Ruff, mypy, all three import contracts, lock validation, build, and diff checks.
- 2026-09-22: Independent review found temporal default overlap, forged-fold replay, order-sensitive derived identity, extra-role coverage, unsafe header lines, quadratic family validation, non-finite metadata hashing, lax temporal parameters, a reserved-name collision, and the documented string-path mismatch. All were reproduced and fixed with regressions.
- 2026-09-22: Fix verification caught an in-band non-finite hash sentinel and extra temporal span roles; replaced the sentinel with recursive type tagging and required temporal spans to match governed roles exactly.
- 2026-09-22: Reviewed revision passed 749 tests (3 deselected), Ruff, mypy, all three import contracts, lock validation, build, and diff checks.

### Completion Notes List

- `dsio.data.splits.generate()` now owns all five approved strategies behind one explicit, non-registerable dispatcher.
- `SplitFile` remains the single persisted contract and records exact assignments, normalized inputs, dependency provenance, declared validations, and a verified content digest.
- Concrete-source validation proves identity membership, coverage, role/group separation, temporal-bound agreement, counts, and cross-fold evaluation uniqueness.
- Stable sample identity is aligned and preserved across shipped `Examples` implementations and subsets.
- Review fixes enforce canonical manifest folds during replay, validate each family once, make temporal defaults and parameters safe, and preserve deterministic identity for reordered and non-finite table metadata.

### File List

- `_bmad-output/implementation-artifacts/2-3-generate-a-governed-split-manifest.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `README.md`
- `docs/adr/0006-splits-are-committed-group-lists.md`
- `pyproject.toml`
- `uv.lock`
- `src/dsio/data/adapters.py`
- `src/dsio/data/examples.py`
- `src/dsio/data/splits/`
- `src/dsio/model/components.py`
- `src/dsio/testing/examples_contract.py`
- `src/dsio/train/ssl_task.py`
- `src/dsio/train/torch_task.py`
- `src/dsio/train/tracking.py`
- `tests/conftest.py`
- `tests/data/test_derivation.py`
- `tests/data/test_examples.py`
- `tests/data/test_split_generation.py`
- `tests/data/test_splits.py`
- `tests/data/test_temporal.py`
- `tests/dataset/test_dataset.py`
- `tests/dataset/test_fixed_size_items.py`
- `tests/eval/test_folds_from_splits.py`
- `tests/model/test_token_corpus.py`
- `tests/test_import_contracts.py`
- `tests/train/test_execute_seeding.py`
- `tests/train/test_ssl_runner.py`
- `tests/train/test_token_run.py`
- `tests/train/test_torch_runner.py`
- Deleted canonical predecessor: `src/dsio/splits/`

### Change Log

- 2026-09-22: Created Story 2.3 and started implementation.
- 2026-09-22: Added governed split generation, exact replay evidence, and the `dsio.data.splits` package migration; candidate passed the complete local quality gate.
