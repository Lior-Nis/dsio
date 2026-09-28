---
title: 'Validate Parkinson full-directory scale behavior'
type: 'feature'
created: '2026-09-27'
status: 'done'
baseline_commit: '5d34627a486777e2e40e29a279fb8670de4c95f8'
context:
  - '{project-root}/CLAUDE.md'
  - '{project-root}/docs/research/2026-09-25-kaggle-gap-portfolio.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The Parkinson reference is proven only on a 134 MiB selection. It neither rejects incomplete official corpora nor reads the 68.34 GB daily-living Parquet lane, and it records no throughput, peak memory, or accelerator-utilization evidence.

**Approach:** Extend the project-owned consumer with a complete official-source inventory, bounded iteration of every non-supervised source file, and phase-specific telemetry. Run the existing supervised experiment over all 924 labelled recordings on CUDA, while describing the result precisely as full labelled training plus full-directory bounded iteration.

## Boundaries & Constraints

**Always:** Resolve official identities from the competition metadata and filter the central source roots against them; fail before benchmarking when an official member is missing, duplicated, or ambiguous. Keep supervised training limited to labelled defog/tdcsfog recordings and preserve test-subject exclusion, score masks, subject-disjoint splitting, exact `DsioModule`/`DsioDataModule`, immutable inference, and MLflow lineage. Read all 46 non-task CSVs and 65 daily Parquets in bounded batches without copying or converting them. Record measurement method, cache state, elapsed time, files, bytes, rows/windows, throughput, process-tree peak RSS, and CUDA utilization/allocated/reserved VRAM. A requested CUDA scale run must fail rather than silently fall back to CPU.

**Ask First:** Any DSIO-core or public-interface change; changing split/model/objective semantics; restaging the daily corpus; introducing a runtime dependency beyond the existing `pyarrow` extra; or widening the work into self-supervised learning, DDP, or model research.

**Never:** Claim the model trained on 70 GB; treat presence as proof of iteration; substitute Forge-derived Zarr for source evidence; import benchmark-test internals into production code; add a generic telemetry framework, streaming abstraction, Kaggle runner, result model, or deployment concern; commit official data or credentials.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Complete central view | Official metadata plus central labelled/daily roots and unrelated extras | Exactly the official 924 labelled, 46 non-task, 65 daily, and two test files are resolved | Extras are ignored and recorded |
| Incomplete or ambiguous view | Missing or duplicate official identity | No scan, staging, or timing evidence begins | Error identifies identity and source lane |
| Non-supervised scale scan | Large CSV/Parquet files with valid schemas | Every row is read in bounded batches; deterministic manifest/checksum and resource evidence are logged | Empty, malformed, unreadable, or schema-drifted files fail with source context |
| Supervised scale run | Full labelled corpus on requested CUDA device | Existing DAG trains, exports, evaluates, and restores all test rows | CUDA absence/OOM or resource-budget breach fails visibly |
| Contract fixture | Tiny source-shaped files without scale roots | Existing deterministic flow behavior remains unchanged | No fabricated scale claim or telemetry |

</frozen-after-approval>

## Code Map

- `reference_projects/kaggle/parkinsons_fog/data.py` -- strict official metadata and labelled-source resolution.
- `reference_projects/kaggle/parkinsons_fog/scale/` -- project-local inventory, bounded scanners, and telemetry.
- `reference_projects/kaggle/parkinsons_fog/tasks/` -- MLflow attempts for inventory, ingestion, training, and downstream evidence.
- `reference_projects/kaggle/parkinsons_fog/flow.py` -- explicit source roots and scale configuration at the project DAG boundary.
- `tests/kaggle_portfolio/test_parkinsons_fog.py` -- contract, completeness, telemetry, and replay coverage.
- `reference_projects/kaggle/README.md`, `docs/research/2026-09-25-kaggle-gap-portfolio.md` -- honest official-run evidence.

## Tasks & Acceptance

**Execution:**
- [x] `reference_projects/kaggle/parkinsons_fog/scale/{inventory,scanning,telemetry}.py` -- resolve a deterministic complete manifest, iterate non-supervised sources in bounded batches, and measure phase resources without a DSIO abstraction.
- [x] `reference_projects/kaggle/parkinsons_fog/data.py`, `tasks/data.py` -- consume the resolved labelled paths, enforce completeness, and log physical/staged counts plus scan evidence.
- [x] `reference_projects/kaggle/parkinsons_fog/tasks/training.py`, `flow.py` -- accept an explicit existing `TrainerConfig`, measure process-tree/CUDA behavior, and expose scale evidence without changing the default contract flow.
- [x] `tests/kaggle_portfolio/test_parkinsons_fog.py`, `conftest.py` -- test missing/duplicate/extra identities, bounded CSV/Parquet traversal, deterministic checksums, telemetry shape, exact classes, and unchanged fixture replay.
- [x] `reference_projects/kaggle/README.md`, `docs/research/2026-09-25-kaggle-gap-portfolio.md` -- record only verified run IDs, metrics, counts, resource envelope, limitations, and the precise claim.
- [x] Central official sources -- run the full scan and one full labelled CUDA experiment in an isolated consumer environment; inspect MLflow in the existing Tailnet UI.

**Acceptance Criteria:**
- Given the central sources, when inventory resolves, then it selects exactly 91 defog and 833 tdcsfog train recordings, 46 non-task CSVs, 65 daily Parquets, and two tests while rejecting missing or ambiguous official IDs.
- Given the full non-supervised lanes, when scanning completes, then all 68,337,598,024 daily-Parquet bytes and every non-task CSV row were read, the checksum is deterministic, batch bounds are retained, and peak scanner RSS remains below 4 GiB.
- Given the full labelled view, when the CUDA flow completes, then all eligible recordings pass through the exact DSIO spine, all 286,370 test predictions regain official order, and MLflow records throughput, peak process-tree RAM, GPU utilization, peak VRAM, lineage, metrics, model, and submission.
- Given the completed benchmark, when documentation is reviewed, then it says “full labelled training plus full-directory bounded iteration” and makes no 70 GB training, DDP, or production-performance claim.

## Spec Change Log

## Design Notes

The 70.59 GB manifest is dominated by 65 unlabeled daily Parquets. Those files are scale evidence, not supervised examples. Scanning computes bounded deterministic evidence and never creates a second large store. Telemetry remains consumer-local until another unrelated scale experiment proves the same interface.

## Verification

**Commands:**
- `uv run pytest -q tests/kaggle_portfolio/test_parkinsons_fog.py` -- contract and scale-boundary tests pass.
- `uv run ruff check . && uv run mypy src && uv run pytest -q` -- repository gates pass.
- Run `parkinsons_fog_flow` with explicit central roots and CUDA `TrainerConfig` -- complete MLflow evidence is browser-visible and resource budgets pass.

## Suggested Review Order

**Scale composition**

- Start with the existing DAG's explicit, optional scale boundary and unchanged default path.
  [`flow.py:24`](../../reference_projects/kaggle/parkinsons_fog/flow.py#L24)

- The scan is a separate MLflow attempt, keeping unlabelled evidence out of training.
  [`data.py:43`](../../reference_projects/kaggle/parkinsons_fog/tasks/data.py#L43)

**Source completeness and bounded iteration**

- Official metadata governs exact source selection while unrelated central extras remain recorded.
  [`inventory.py:22`](../../reference_projects/kaggle/parkinsons_fog/scale/inventory.py#L22)

- Labelled loading revalidates inventory fingerprints immediately before reading each source.
  [`data.py:25`](../../reference_projects/kaggle/parkinsons_fog/data.py#L25)

- Non-supervised scanning validates stable sources and hashes bounded logical batches.
  [`scanning.py:29`](../../reference_projects/kaggle/parkinsons_fog/scale/scanning.py#L29)

**Resource evidence and training**

- Phase measurement records explicit methods, sampled memory, and physical CUDA identity.
  [`telemetry.py:18`](../../reference_projects/kaggle/parkinsons_fog/scale/telemetry.py#L18)

- All-thread recursive child traversal captures Prefect and DataLoader worker memory.
  [`telemetry.py:184`](../../reference_projects/kaggle/parkinsons_fog/scale/telemetry.py#L184)

- Fit-only timing and completed-epoch throughput preserve exact DSIO Lightning execution.
  [`training.py:58`](../../reference_projects/kaggle/parkinsons_fog/tasks/training.py#L58)

**Acceptance proof**

- Scale-flow tests prove explicit configuration, evidence linkage, and unchanged contract behavior.
  [`test_parkinsons_fog.py:490`](../../tests/kaggle_portfolio/test_parkinsons_fog.py#L490)

- Process and scanner tests cover bounded reads, mutation, shutdown, and worker-memory boundaries.
  [`test_parkinsons_fog.py:265`](../../tests/kaggle_portfolio/test_parkinsons_fog.py#L265)

- Browser-visible run IDs and limitations make the final scale claim independently auditable.
  [`README.md:75`](../../reference_projects/kaggle/README.md#L75)
