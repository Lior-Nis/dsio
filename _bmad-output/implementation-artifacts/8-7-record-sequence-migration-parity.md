---
baseline_commit: d26ae16ceb01f34af845fb32efcfdccb4c69b8c4
---

# Story 8.7: Record sequence migration parity

Status: done

## Story

As a maintainer,
I want representative-tier parity for the migrated sequence consumers recorded in MLflow,
so that Epic 8 proves behavior preservation at real-data scale.

## Acceptance Criteria

1. Given the immutable Story 6.6 baseline metrics, when Essay Scoring, Parkinson FoG, and ROGII rerun on the recorded CPU/12-thread hardware class, then every metric is compared under the explicit comparison rule below, the baseline metric values remain unchanged, and any mismatch reports the complete signed delta.
2. Given a successful comparison, when its evidence is inspected in MLflow, then a dedicated parity run identifies the baseline and observed evaluation runs and records baseline, observed, delta, tolerance, and pass status for every metric.
3. Given every warehouse block used by the three consumers, when the evidence register is checked, then that consumer use records the new immutable tailnet parity-run URI; catalog and admission checks pass.
4. Given the completed Epic 8 migration, when consumer metrics are measured, then the three consumers have no uncatalogued local dataset, collator, model, objective, normalizer, or validator definition; their non-ingestion line delta is recorded; all repository gates pass; and Epic 8 closes.

## Comparison Rule

- No Epic 8 story declared a representative metric tolerance. All three consumers therefore require absolute tolerance zero from a clean tree under Python 3.12 on the baseline CPU/12-thread hardware class.
- The first clean FoG comparison failed at zero. Investigation corrected a real float32
  reduction-order defect, then isolated the metric divergence to legacy convolutional
  boundary arithmetic across structural padding. Both failed comparisons remain immutable
  diagnostic evidence; the final comparison passes at zero.
- Preserve every baseline metric and `tests/golden_metrics.json`. The three sequence baseline
  run IDs were replaced only after review proved their MLflow provenance was dirty despite
  the baseline records saying otherwise; clean reruns at the same historical commit reproduced
  every metric exactly.

## Tasks / Subtasks

- [x] Preflight all three immutable baseline evaluation runs and confirm the clean committed source, Python 3.12, CPU identity, and 12 Torch threads (AC 1).
- [x] Run `essay_scoring`, `parkinsons_fog`, and `rogii` through the existing representative comparison path at tolerance zero; verify the dedicated MLflow parity runs and complete deltas (AC 1-2).
- [x] Attach each parity-run URI to every warehouse block used by its consumer, regenerate the catalog, and leave fixture-only evidence unchanged (AC 3).
- [x] Measure the three consumers from the Story 8.1 baseline; confirm those consumers have zero banned definitions, with repository residuals limited to 10 registered Epic 9 candidates and 3 explicit self-supervised fixtures (AC 4).
- [x] Run focused consumer/parity checks, full pytest, Ruff, format, mypy, import contracts, catalog/admission, build, distribution/consumer-flow contracts, and golden/baseline integrity checks (AC 1-4).
- [x] Run the three-layer adversarial review, resolve actionable findings, mark Story 8.7 and Epic 8 done, and merge through CI (AC 1-4).

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

- Rejected baseline evaluations: Essay `bfeccfe3d94b418d9b8b9db9f06440eb`, FoG
  `083b5f32aba04d01b938532c10ee029e`, and ROGII
  `35d889f9dd3d4b6ebaeb04c4606b3d2b` all recorded dirty MLflow provenance.
- Clean historical replacements: Essay `47beb58ce2df4badae0d1a62a5d6f4d0`, FoG
  `576d18d283d746aeba3b0b27e7351974`, and ROGII
  `ee7da6013406443b9cb1482d67fbeda6`; all metrics were exact reproductions.
- Failed zero-tolerance FoG diagnostic: parity `43e8daf138b34320ad9a200d4294a71d`,
  evaluation `36851b3927a84a8da4012a87e665d6b3`.
- Rejected post-observation tolerance evidence was never attached to the final register. A
  later zero-tolerance diagnostic after the reduction fix, parity
  `06d22f1f027f486d86da86df6eee46b6`, proved normalization was not the metric cause.
- Final clean zero-tolerance parity: Essay `71a7cb3803674fb9986fda1928b597e6`,
  FoG `5dddb9d97cc04e6ca725a7dfaa03fb1a`, and ROGII
  `69a9ad9f8c6948829f938654b6d8c9a1`. All are `FINISHED`, tagged `passed`,
  use tolerance `0.0`, and have maximum absolute delta `0.0`.
- Dirty diagnostic `f8e62a709ff547e997e6872db5aa431f` tested native Torch
  normalization with legacy boundary arithmetic. Its AP values moved by at most
  `3.1e-9`, proving the explicit NumPy reduction remains necessary for literal equality.

### Completion Notes List

- Recovered the three sequence baselines from clean historical commit `fe37300c844d`.
  Every metric reproduced exactly; only the invalid MLflow run identities changed.
- Hardened representative preflight to download `provenance.json`, reject dirty baseline
  evidence, and require its Git SHA to match the baseline record.
- Added explicit compatibility modes without weakening safe defaults:
  `InstanceStandardize(reduction="numpy")` for bit-exact audited NumPy migrations and
  `DenseConv1d(isolate_padding=False)` for legacy boundary arithmetic. FoG alone opts in.
- Three-layer review retained those opt-ins as an explicit consequence of the owner's
  zero-tolerance choice: FoG's legacy mode is CPU-oriented and batch-boundary-dependent;
  the reusable component defaults remain GPU-native, differentiable, and padding-isolated.
  A later behavior-change story may remove the opt-ins, but cannot call that change parity.
- Review hardened NumPy tiny-epsilon handling and baseline preflight: evaluation runs must
  be `FINISHED`, Git-clean at the recorded SHA, match Python/Torch runtime, and expose the
  exact committed metrics. The temporary recovery launcher added no behavior; it only
  called the versioned historical `tools.representative.run` function sequentially.
- Attached final parity evidence to 8 Essay uses, 17 FoG uses, and 8 ROGII uses; regenerated
  catalog validation passes.
- The three consumers contain zero banned local definitions. Non-ingestion code is 1,765
  lines (Essay 502, FoG 736, ROGII 527), 75 fewer than the Story 8.1 baseline of 1,840.
  Repository residuals are 10 registered Epic 9 CMI candidates plus 3 explicit
  `self_supervised` fixtures.
- Verification: 47 focused tests passed before the final boundary correction; final build
  passed, 8 distribution/consumer-flow contracts passed, 1,319 main tests passed (1 live
  test deselected), Ruff lint/format passed, mypy passed for 124 source files, all 3 import
  contracts held, and catalog/admission checks passed. Golden metrics were unchanged;
  recovered baseline metric values were unchanged.

### File List

- `_bmad-output/implementation-artifacts/8-3-compose-dense-signal-models-with-layout-adapters-and-per-instance-standardization.md`
- `_bmad-output/implementation-artifacts/8-7-record-sequence-migration-parity.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `docs/component-warehouse/catalog.md`
- `docs/component-warehouse/evidence.yaml`
- `docs/component-warehouse/parity-baseline.json`
- `docs/component-warehouse/parity-baseline.md`
- `reference_projects/kaggle/parkinsons_fog/components.py`
- `src/dsio/experimental/model/convolution.py`
- `src/dsio/experimental/model/standardization.py`
- `tests/experimental/model/test_signals.py`
- `tests/test_representative.py`
- `tools/representative.py`

## Change Log

- 2026-10-03: Created Story 8.7 as a minimal evidence-only closure of Epic 8.
- 2026-10-03: Recovered clean historical baselines, hardened provenance preflight, and added an
  explicit NumPy-exact reduction for parity-constrained migrations.
- 2026-10-04: Isolated FoG divergence to legacy padded-boundary convolution, recorded exact
  parity for all three sequence consumers, attached evidence, and completed repository gates.
