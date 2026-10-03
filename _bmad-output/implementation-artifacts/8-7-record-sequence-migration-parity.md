---
baseline_commit: d26ae16ceb01f34af845fb32efcfdccb4c69b8c4
---

# Story 8.7: Record sequence migration parity

Status: in-progress

## Story

As a maintainer,
I want representative-tier parity for the migrated sequence consumers recorded in MLflow,
so that Epic 8 proves behavior preservation at real-data scale.

## Acceptance Criteria

1. Given the immutable Story 6.6 baseline, when Essay Scoring, Parkinson FoG, and ROGII rerun on the recorded CPU/12-thread hardware class, then every metric is compared under the tolerance declared before its migration, the baseline remains unchanged, and any mismatch reports the complete signed delta.
2. Given a successful comparison, when its evidence is inspected in MLflow, then a dedicated parity run identifies the baseline and observed evaluation runs and records baseline, observed, delta, tolerance, and pass status for every metric.
3. Given every warehouse block used by the three consumers, when the evidence register is checked, then that consumer use records the new immutable tailnet parity-run URI; catalog and admission checks pass.
4. Given the completed Epic 8 migration, when consumer metrics are measured, then the three consumers have no uncatalogued local dataset, collator, model, objective, normalizer, or validator definition; their non-ingestion line delta is recorded; all repository gates pass; and Epic 8 closes.

## Predeclared Comparison Rule

- No Epic 8 story declared a representative metric change. Run all three comparisons at absolute tolerance zero from a clean tree under Python 3.12 on the baseline CPU/12-thread hardware class.
- Story 8.3's `rtol=1e-6`, `atol=1e-6` allowance is limited to the moved per-instance standardization tensor computation; it does not authorize selecting a wider representative metric tolerance after observing a result.
- Do not modify `docs/component-warehouse/parity-baseline.json`, its rendered report, or `tests/golden_metrics.json`.

## Tasks / Subtasks

- [ ] Preflight all three immutable baseline evaluation runs and confirm the clean committed source, Python 3.12, CPU identity, and 12 Torch threads (AC 1).
- [ ] Run `essay_scoring`, `parkinsons_fog`, and `rogii` through the existing representative comparison path at tolerance zero; verify the dedicated MLflow parity runs and complete deltas (AC 1-2).
- [ ] Attach each parity-run URI to every warehouse block used by its consumer, regenerate the catalog, and leave fixture-only evidence unchanged (AC 3).
- [ ] Measure the three consumers from the Story 8.1 baseline and confirm all remaining banned definitions belong only to registered later-epic candidates (AC 4).
- [ ] Run focused consumer/parity checks, full pytest, Ruff, format, mypy, import contracts, catalog/admission, build, distribution/consumer-flow contracts, and golden/baseline digest checks (AC 1-4).
- [ ] Run the three-layer adversarial review, resolve actionable findings, mark Story 8.7 and Epic 8 done, and merge through CI (AC 1-4).

## Dev Notes

- Reuse `tools/representative.py --compare`; do not add another parity command, result object, report hierarchy, or DSio API.
- The Story 8.6 full-data runs were dirty-tree implementation evidence, not parity certification. This story must create clean-commit comparison and parity runs.
- MLflow remains the source of truth. Evidence uses browser-visible `https://pop.tailee691f.ts.net:8443` run URIs; API calls use `http://127.0.0.1:5000`.
- If an exact comparison fails, investigate reproducibility before changing code. A post-observation tolerance is forbidden.
- Operational run workspaces stay outside the repository under `/home/liornisimov/Projects/dsio-runs/story-8-7`.

### References

- [Source: `_bmad-output/planning-artifacts/epics-component-warehouse.md` — Story 8.7 and delivery rules]
- [Source: `_bmad-output/implementation-artifacts/7-7-record-tabular-migration-parity.md`]
- [Source: `_bmad-output/implementation-artifacts/8-3-compose-dense-signal-models-with-layout-adapters-and-per-instance-standardization.md`]
- [Source: `_bmad-output/implementation-artifacts/8-6-declare-dense-and-ordinal-prediction-outputs.md`]
- [Source: `tools/representative.py`]
- [Source: `docs/component-warehouse/parity-baseline.json`]
- [Source: `docs/component-warehouse/evidence.yaml`]

## Dev Agent Record

### Agent Model Used

GPT-5

### Debug Log References

- Pending.

### Completion Notes List

- Pending.

### File List

- Pending.

## Change Log

- 2026-10-03: Created Story 8.7 as a minimal evidence-only closure of Epic 8.
