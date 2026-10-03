---
baseline_commit: 6db3eea1538de68106e9199ce8dbf134239a9150
---

# Story 7.7: Record tabular migration parity

Status: review

## Story

As a maintainer,
I want representative-tier parity for the migrated tabular consumers recorded in MLflow,
so that Epic 7 proves the warehouse preserved behavior at real-data scale.

## Acceptance Criteria

1. Given the immutable Story 6.6 baseline, when Titanic, Bike Sharing, Store Sales and Digit Recognizer rerun on the same recorded hardware class, then each migrated metric is compared with absolute tolerance zero, the baseline is not overwritten, and any mismatch fails the command with the complete metric delta.
2. Given one successful comparison, when its evidence is inspected in MLflow, then a dedicated parity run identifies the baseline and observed evaluation runs, records the baseline/observed/delta metrics and zero tolerance, and is reachable through an immutable tailnet run URI.
3. Given every warehouse block used by a migrated tabular consumer, when the evidence register is checked, then that consumer's use records the new parity run URI; catalog and admission checks pass.
4. Given the Epic 7 migration, when the consumer measurement tool runs, then the four tabular consumers have no uncatalogued local model, objective, dataset, normalizer, validator or collation finding, and their non-ingestion line delta from Story 6.6 is recorded.

## Tasks / Subtasks

- [x] Task 1: Complete and protect the representative baseline (AC 1).
  - [x] Confirm the baseline entries and raw evaluation run IDs for all four consumers before any migrated run starts.
  - [x] Recover Bike Sharing's missing pre-migration representative baseline by running commit `fe37300c844de7db0c54189ec0c45b9c16cdd101` on the recorded CPU class; never substitute a post-migration run as its baseline.
  - [x] Download the Bike Sharing competition files into `~/Datasets/dsio-kaggle-portfolio/bike-sharing` if absent; keep datasets outside the repository.
  - [x] Add a regression that comparison mode cannot mutate `docs/component-warehouse/parity-baseline.json` or its rendered baseline report.
- [x] Task 2: Record generic MLflow parity evidence in `tools/representative.py` (AC 1, 2).
  - [x] On `--compare`, validate the observed hardware class against the baseline and compare every metric using the requested tolerance.
  - [x] Create one dedicated MLflow parity run in the observed evaluation experiment. Tag it with consumer, baseline commit/run ID, observed commit/run ID, tolerance and pass/fail status; log baseline, observed and signed delta values for every metric.
  - [x] Preserve raw run IDs separately from browser URIs. Use the live tracking endpoint for MLflow API calls and a configurable HTTP(S) UI base for committed links; never persist credentials or a file-store URI.
  - [x] Print the parity run ID/URI and complete deltas for evidence registration. MLflow and `evidence.yaml` are the sources of truth; do not add a parallel result schema or rewrite the Story 6.6 baseline.
  - [x] Add focused tests for exact parity, mismatches, missing metrics, hardware mismatch, nested/multi-model run selection, MLflow tags/metrics/status, UI URI construction and worker isolation.
- [x] Task 3: Run the four real representative consumers (AC 1, 2).
  - [x] Set the live MLflow tracking endpoint and tailnet UI base explicitly; use the runner's isolated Prefect subprocess and fresh central run workspace.
  - [x] Run `titanic`, `bike_sharing`, `store_sales` and `digit_recognizer` with seed 19 and tolerance zero on the recorded 12-thread CPU class.
  - [x] Verify every comparison passes exactly. If any metric moves, stop: no tolerance was pre-declared in Stories 7.1-7.6, so this story may not invent one after observing a delta.
  - [x] Verify the parity runs and their metrics are visible at `https://pop.tailee691f.ts.net:8443`.
- [x] Task 4: Attach the real evidence and record deletion (AC 3, 4).
  - [x] Add each consumer's parity run URI to every corresponding real-use record for `StoredItems`, fitting/standardization, compositions, objectives, outputs/validators and `collate_arrays`; keep fixture uses unchanged.
  - [x] Regenerate `docs/component-warehouse/catalog.md` and verify every added URI has a matching consumer use.
  - [x] Run `tools/consumer_metrics.py`; record the four-consumer baseline/current non-ingestion totals, delta and remaining banned definitions in the Dev Agent Record and PR.
- [x] Task 5: Close Epic 7 with full verification (AC 1-4).
  - [x] Run the four contract-tier consumer suites without editing `tests/golden_metrics.json`.
  - [x] Run the full test, static, import-contract, catalog, admission and distribution gates.
  - [x] Confirm baseline metric values and all contract-tier goldens are byte-for-byte unchanged.

## Dev Notes

- `tools/representative.py` already runs each consumer in a fresh process with `PREFECT_*` isolation, preserves raw run IDs and resolves UI links. Extend that one runner; do not add a second parity CLI or a result dataclass.
- Comparison evidence is new output. `docs/component-warehouse/parity-baseline.{json,md}` remains the immutable pre-migration input; use a separate `parity-runs` pair for migrated results.
- The current baseline contains Titanic, Store Sales and Digit Recognizer, but no Bike Sharing entry. The only valid recovery is a real run from the recorded pre-migration commit, not a current-code estimate or contract-tier golden.
- Baseline hardware for the tabular entries is `AMD Ryzen 9 9900X 12-Core Processor`, CPU, 12 Torch threads. Hardware class means accelerator class plus device/CPU model and configured thread count; Python/Torch versions are recorded context, not equality gates.
- Use Python 3.12 for the representative reruns to match the recorded environment; the default `uv` interpreter may otherwise select Python 3.13.
- The baseline evaluation run is the `evaluation_run_id` in each baseline record. The observed evaluation run comes from the migrated flow result. A parity run is separate from both and must remain inspectable even when comparison fails.
- Metric logging should use stable namespaces such as `baseline.<metric>`, `observed.<metric>` and `delta.<metric>`; do not hide missing metrics or coerce them to zero.
- The four real-data runs are acceptance evidence, not mocked tests. Keep their workspaces under `~/Datasets/dsio-runs/representative`; do not write generated data into the Git worktree.
- No new DSio package API, training abstraction, deployment behavior or task-specific shared-code branch belongs in this story.

### Project Structure Notes

- Expected repository changes: `tools/representative.py`, `tests/test_representative.py`, `docs/component-warehouse/{parity-baseline.json,parity-baseline.md,evidence.yaml,catalog.md}`, this story file and sprint status. The baseline pair changes only to add the recovered historical Bike entry.
- A historical detached worktree and downloaded Bike data are operational inputs only and must not appear in the commit.
- Update consumer code only if a real run exposes a defect caused by Stories 7.1-7.6; add a regression first and preserve exact goldens.

### Previous Story Intelligence

- Story 7.6 moved downstream arrays through `collate_arrays`; the subsequent hardening pass added ordered identity guarding, exact dataset/collator provenance and isolated representative workers.
- Representative worker results intentionally project only metrics and run IDs because consumer results contain submission bytes.
- Run IDs and run URIs are distinct fields. Evidence uses immutable HTTP(S) run URIs; MLflow API calls use raw IDs.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 7.7 and delivery rules]
- [Source: `_bmad-output/specs/spec-component-warehouse/SPEC.md` — CAP-10, determinism and evidence constraints]
- [Source: `_bmad-output/specs/spec-component-warehouse/conventions.md` — evidence register]
- [Source: `_bmad-output/implementation-artifacts/7-6-build-evaluation-arrays-from-the-training-collation.md`]
- [Source: `tools/representative.py`]
- [Source: `tools/consumer_metrics.py`]
- [Source: `docs/component-warehouse/parity-baseline.json`]
- [Source: `docs/component-warehouse/evidence.yaml`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Implementation Plan

- Recover the missing Bike baseline from the historical commit before current-code execution.
- Extend the existing representative runner with MLflow-native comparison evidence.
- Run the four consumers at exact tolerance, attach their parity URIs, and close all gates.

### Debug Log References

- Bike Sharing historical run: evaluation `8f38a18dd6744f95afe9215bb8b70f3d`, RMSE `204.82968446829918`.
- Parity runs: Titanic `a0e83b068f4c4146a3d2158bcb933721`; Bike `6c37805abd37412f99859815517ce85d`; Store Sales `6df24c8d13a14d5bb1b73459f5d2d22f`; Digit `990bcbd49b1147d0a04ecf0af98459e9`.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Recovered the Bike Sharing baseline from `fe37300c844de7db0c54189ec0c45b9c16cdd101` on the recorded CPU/12-thread hardware class.
- Added immutable comparison-mode protection and MLflow parity-run coverage without introducing a parallel result model.
- All four migrated consumers matched their real-data baselines exactly at absolute tolerance zero; their MLflow parity runs are `FINISHED` with zero deltas.
- Four-consumer non-ingestion code fell from 2,207 lines at the Story 6.6 baseline to 2,184 lines now (`-23`); none of the four consumers has a remaining banned local definition.
- Verification completed: four consumer suites `22 passed`; full suite `1183 passed, 1 deselected`; Ruff, format, mypy, import contracts, catalog, admission, build and distribution contracts all pass.
- Golden metrics and the completed four-consumer baseline remained byte-for-byte unchanged throughout migrated execution and final verification.

### File List

- `_bmad-output/implementation-artifacts/7-7-record-tabular-migration-parity.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/parity-baseline.json`
- `docs/component-warehouse/parity-baseline.md`
- `docs/component-warehouse/evidence.yaml`
- `docs/component-warehouse/catalog.md`
- `tests/test_representative.py`
- `tools/representative.py`

## Change Log

- 2026-10-03: Recovered Bike baseline and added generic MLflow parity recording.
- 2026-10-03: Recorded exact real-data parity, attached evidence, and measured Epic 7 deletion.
