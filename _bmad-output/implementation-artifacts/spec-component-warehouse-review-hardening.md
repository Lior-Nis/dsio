---
title: 'Harden the component warehouse after Stories 7.1-7.6'
type: 'bugfix'
created: '2026-10-03'
status: 'done'
baseline_commit: '802761b7b976550cdc10303c390d9aa7d0cfd53d'
context:
  - '_bmad-output/specs/spec-component-warehouse/SPEC.md'
  - '_bmad-output/specs/spec-component-warehouse/conventions.md'
  - '_bmad-output/specs/spec-component-warehouse/component-cohort.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The Story 7.1-7.6 review found contract holes in identity guarding, training-only fitting, provenance, accumulation semantics, catalog governance, and representative evidence. These defects can silently admit leakage, change optimization, or produce evidence that cannot be reproduced.

**Approach:** Close each demonstrated hole at its narrowest owning boundary, delete superseded APIs, and keep consumer configurations aligned with the DSio contracts without introducing new result models or parallel abstractions.

## Boundaries & Constraints

**Always:** Preserve ordered sample identity; bind fitted values to the manifest's store and declared training role; include dataset and collation configuration in execution identity; keep optimization invariant to uneven gradient-accumulation partitions; isolate representative runs before importing Prefect consumers; require a live HTTP(S) MLflow server; reject duplicate governance keys; retain exact contract-tier parity.

**Ask First:** Any fix that changes a recorded golden metric, adds a public abstraction rather than reusing an existing one, or requires re-recording representative baselines for reasons other than run-URI normalization.

**Never:** Preserve dead APIs for backwards compatibility; move project deployment or promotion policy into DSio; add task-specific branches to shared code; weaken validation to make existing consumers pass.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Fit statistics | Manifest and its matching store | Only the declared training assignment is read | Reject a different store, missing role, or empty assignment before reading samples |
| Collate arrays | Ordered IDs and a map-style dataset | Arrays preserve the requested ID at every position | Reject reversed IDs, wrong length, iterable datasets, and non-mapping items |
| Accumulate gradients | Same effective batch split into uneven microbatches | One optimizer step matches the unsplit batch | Fail fast for unsupported trainer modes rather than silently mis-scale |
| Representative run | Live MLflow URI and one consumer | Consumer imports and runs inside an isolated Prefect process; evidence contains immutable run URIs | Reject file-backed/non-HTTP tracking and propagate worker failure |
| Catalog inputs | Stable module or governance YAML | Public components and every YAML entry are checked exactly once | Reject undeclared public APIs and duplicate YAML keys |

</frozen-after-approval>

## Code Map

- `src/dsio/experimental/data/fitting.py` -- training-only, store-bound statistic fitting.
- `src/dsio/experimental/data/arrays.py` -- evaluation/inference collation with identity guards.
- `src/dsio/model/module.py` -- Lightning optimization and gradient accumulation boundary.
- `tools/representative.py` -- isolated real-consumer execution and MLflow evidence.
- `tools/catalog.py` -- component discovery and governance YAML loading.
- `reference_projects/**/components.py` -- canonical dataset/preprocessing declarations.
- `reference_projects/**/tasks/downstream.py` -- evaluation/export/inference provenance.
- `docs/component-warehouse/{candidates.yaml,evidence.yaml,parity-baseline.*}` -- governed evidence.

## Tasks & Acceptance

**Execution:**
- [x] Add regression tests for every reproduced contract failure before changing its owner.
- [x] Harden fitting and array collation at the DSio data boundaries.
- [x] make `DsioModule` normalize automatic gradient accumulation by samples, including uneven final windows.
- [x] Align migrated consumer dataset/preprocessing configuration and record dataset/collator provenance.
- [x] Run representative consumers in a fresh process with pre-import Prefect isolation, live tracking validation, and immutable run URIs.
- [x] Close catalog/YAML bypasses, merge the duplicate candidate, and remove `bce_loss`/`mse_loss` completely.
- [x] Update generated catalog and review documentation that became inaccurate.

**Acceptance Criteria:**
- Given each review reproducer, when run against the fix, then it either produces the contractually correct result or raises a specific boundary error.
- Given the existing contract-tier flows, when replayed, then all committed goldens remain exact.
- Given the full quality gate, when run from a clean checkout, then tests, type checks, lint, package build, catalog check, and admission audit pass.

## Spec Change Log

- 2026-10-03: Implemented every hardening task regression-first; normalized representative
  run references without changing recorded metrics; corrected the admission verification
  command to the repository's test-based audit.
- 2026-10-03: Adversarial review closed mutable-identity, worker-serialization, catalog,
  training-collator provenance, and run-ID/URI gaps. Preserved the manifest's positional
  training-role contract and made unsupported accumulation modes fail explicitly.

## Design Notes

Sample-count normalization belongs in the one reusable Lightning module because objectives should describe per-batch loss semantics, not know trainer accumulation windows. Dataset identity guarding reuses `IdentityDataset`; governance parsing remains a repository-tool concern.

Automatic accumulation is normalized immediately before the optimizer step from the actual
number of samples observed in that window. A private capability marks objectives whose loss
is provably a sample mean; accumulation greater than one fails for every other objective.
Distributed, strategy-owned, and closure-replaying accumulation modes also fail explicitly
because DSio cannot preserve their sample semantics here.

## Verification

**Commands:**
- `uv run pytest -q` -- all tests pass.
- `uv run mypy && uv run ruff check . && uv run ruff format --check .` -- static gates pass.
- `uv run python tools/catalog.py --check && uv run pytest -q tests/experimental/test_admission.py tests/experimental/test_admission_coverage.py` -- governance gates pass.
- `uv build` -- wheel and source distribution build successfully.

## Suggested Review Order

**Optimization semantics**

- Start here: sample-count normalization and explicit unsupported-mode failures.
  [`module.py:211`](../../src/dsio/model/module.py#L211)

- Private capability admits only objectives with a provable sample-mean reduction.
  [`objectives.py:119`](../../src/dsio/experimental/model/objectives.py#L119)

- Masked-point training rejects incompatible calibrated accumulation without changing its loss.
  [`training.py:220`](../../reference_projects/kaggle/parkinsons_fog/tasks/training.py#L220)

**Data boundaries**

- Fitting validates store binding and reads only the manifest-declared training role.
  [`fitting.py:14`](../../src/dsio/experimental/data/fitting.py#L14)

- Array collation snapshots requested identities and reuses the training identity guard.
  [`arrays.py:20`](../../src/dsio/experimental/data/arrays.py#L20)

**Consumer provenance**

- Bike injects fitted statistics through component resolution and records exact inputs.
  [`training.py:58`](../../reference_projects/kaggle/bike_sharing/tasks/training.py#L58)

- Digit owns pixel scaling inside the shared classifier composition exactly once.
  [`components.py:79`](../../reference_projects/kaggle/digit_recognizer/components.py#L79)

**Evidence and governance tooling**

- Representative runs isolate imports, preserve IDs, and add immutable server URIs.
  [`representative.py:99`](../../tools/representative.py#L99)

- Catalog collection enforces declared surfaces and retains every governance failure.
  [`catalog.py:202`](../../tools/catalog.py#L202)

**Regression coverage**

- Lightning integration proves uneven 3/3/2 accumulation matches an unsplit batch.
  [`test_training_spine.py:284`](../../tests/model/test_training_spine.py#L284)

- Worker projection covers real consumer results containing non-JSON submission bytes.
  [`test_representative.py:188`](../../tests/test_representative.py#L188)

- Catalog regression preserves multiple undeclared-name findings per module.
  [`test_component_catalog.py:269`](../../tests/test_component_catalog.py#L269)
