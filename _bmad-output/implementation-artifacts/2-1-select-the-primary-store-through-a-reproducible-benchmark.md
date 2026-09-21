---
baseline_commit: 3bb74c7eec8c059c7e716874032b8e8b0c00fa86
---

# Story 2.1: Select the Primary Store Through a Reproducible Benchmark

Status: review

## Story

As a DSio data practitioner,
I want the primary storage format selected from measured representative workloads,
so that the library makes one evidence-based storage choice instead of exposing speculative backend complexity.

## Acceptance Criteria

1. One checked-in benchmark applies equivalent build, sequential-window, random-window, and multi-worker reads to a focused candidate set on declared representative synthetic profiles.
2. Each result records throughput, elapsed build cost, peak process memory, disk bytes, and objective operational facts, including incomplete-write behavior.
3. The saved machine-readable report records every input, seed, environment fact, candidate package version, and raw measurement needed to interpret the run.
4. The complete synthetic benchmark reruns from one documented command in the locked benchmark environment; a project-owned real corpus can be supplied without embedding a machine-specific path in code.
5. The existing real-FORGE evidence is retained with its measured conditions clearly identified, and the primary-format decision explains workload-specific tradeoffs without universal claims.
6. DSio exposes only the selected flat-binary/memory-mapped implementation: no public backend selector, backend registry, or benchmark abstraction enters `src/dsio`.

## Tasks / Subtasks

- [x] Replace the brittle exploratory scripts with one reproducible benchmark (AC: 1-4)
  - [x] Put benchmark-only code outside the shipped `src/dsio` package and pin its non-runtime dependencies in a dedicated dependency group.
  - [x] Generate at least two deterministic synthetic workload profiles and accept an explicit project-owned real input.
  - [x] Exercise identical materializing reads for every candidate and isolate multi-worker handles per process.
  - [x] Persist structured configuration, environment, versions, raw timings, checksums, disk use, peak RSS, file count, and interrupted-write observations.
- [x] Re-run and document the storage decision (AC: 3-5)
  - [x] Commit a synthetic result from the locked command and a real-corpus result from the available FORGE corpus.
  - [x] Rewrite ADR 0005 around the reproducible evidence, retaining historical results only when labeled as historical.
  - [x] State limits: local filesystem, warm-cache behavior, tested shapes/dtypes, machine, versions, and candidate scope.
- [x] Collapse the production read path to the selected implementation (AC: 6)
  - [x] Remove the backend name parameter and runtime reader registry while preserving per-process lazy memory-map opening.
  - [x] Keep storage validation, stable public imports, and root `import dsio` behavior unchanged.
- [x] Verify and review (AC: 1-6)
  - [x] Add fast tests for benchmark configuration/report completeness and interrupted-write classification; never assert one candidate is faster in CI.
  - [x] Run focused tests, a smoke benchmark, full pytest, Ruff, mypy, import-linter, lock validation, build, and diff checks.
  - [ ] Complete independent blind, edge-case, and acceptance reviews before merge.

## Dev Notes

### Scope and minimal design

- This story decides and records a physical format. Story 2.2 will evolve the stable sample store interface; do not prebuild that interface here.
- Keep the benchmark as ordinary Python functions plus one `python -m` entry point. Do not create candidate protocols, registries, result dataclasses, a CLI framework, or production backend hooks.
- Candidate-specific code belongs only in `benchmarks/storage/`. The benchmark may compare flat binary, Arrow IPC, and Zarr because they span contiguous memory-mapped and chunked/compressed designs; their packages are benchmark dependencies, not DSio runtime dependencies.
- The benchmark report is plain JSON. Reuse dictionaries/lists rather than defining a parallel experiment or result model.
- A performance ordering is evidence, not a test invariant. CI verifies schema, equivalence, determinism of workload generation, and that the command completes at smoke scale.

### Current state to preserve or change

- `SignalStore` writes an immutable C-contiguous `signal.bin`, versioned `signal.idx`, entity metadata, and a digest manifest. Preserve its layout and corruption checks.
- `SignalStore` lazily opens a reader per PID, which prevents a live mapping from being serialized into spawn workers. Preserve this behavior.
- `SignalStore(..., backend=...)`, `SignalStore.open(..., backend=...)`, `readers.BACKENDS`, and `open_reader(backend, ...)` expose a backend seam with only one implementation. Remove that speculative choice and open `MmapReader` directly.
- ADR 0005 and its three Python scripts contain valuable historical FORGE measurements, but the scripts hard-code local paths, undeclared packages, and separate methods. Replace them rather than adding a fourth script.
- The existing FORGE corpus is project-owned. A benchmark input option may read its `accs` array, but DSio must not import FORGE code or assume that path exists.

### Benchmark contract

- Inputs: explicit seed; rows/channels/window/read count/worker counts for each synthetic profile; candidate settings; optional real array path/key/row limit.
- Equivalent operations: materialize and checksum every read; use the same starts and logical values across candidates; reopen candidate handles in each worker.
- Measurements: build elapsed seconds and throughput, sequential and random read elapsed seconds/throughput, multi-worker aggregate throughput, peak RSS, on-disk bytes/file count, and recovery observations after a deliberately incomplete build.
- Environment: timestamp, OS/platform, architecture, Python, CPU count, DSio commit/dirty state, and exact NumPy/PyArrow/Zarr versions.
- Result integrity: record source shape/dtype/digest or explicit source identity and candidate output checksums. Fail the run if candidates return different logical values.
- Operational facts must be observable (dependency/version, file count, configuration choices, incomplete output readability), not an invented aggregate score.

### Project structure

- New benchmark package: `benchmarks/storage/`; split into cohesive files only when a single module becomes difficult to navigate.
- Fast benchmark contract tests: `tests/benchmarks/`.
- Decision and result evidence: `docs/adr/0005-canonical-store-is-flat-binary.md` and a small JSON result directory adjacent to it or under `benchmarks/storage/results/`.
- Delete the obsolete `docs/adr/0005-*.py` scripts after their reproducible replacement exists.

### Testing and completion guardrails

- Tests use tiny arrays and low read counts; they validate correctness, report fields, deterministic workload selection, and failure messages, not timing rank.
- Exercise spawn-safe multi-worker behavior on the supported platform. If a platform cannot report RSS identically, record the method/availability rather than fabricating a value.
- The documented full command must use the lockfile and dedicated benchmark dependency group.
- No runtime dependency, public backend registration, remote-store behavior, or general performance claim is permitted.

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Epic 2, Story 2.1, FR22]
- [Source: `docs/superpowers/specs/2026-09-18-generic-experiment-spine.md` — Data and splits; Package shape]
- [Source: `docs/adr/0005-canonical-store-is-flat-binary.md`]
- [Source: `src/dsio/data/store.py`]
- [Source: `src/dsio/data/readers.py`]
- [NumPy memory mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [Arrow memory-mapped I/O](https://arrow.apache.org/docs/python/memory.html#memory-mapping)
- [Zarr array creation](https://zarr.readthedocs.io/en/main/api/zarr/functions/create_array/)

## Dev Agent Record

### Agent Model Used

Codex (GPT-5)

### Debug Log References

- 2026-09-21: Created from merged Epic 1 at `3bb74c7`; inspected the accepted spine, Epic 2, current store/read path, ADR 0005, historical benchmark scripts, dependency lock, and locally available FORGE corpus.
- 2026-09-21: Full pre-review gate passed with 679 tests, Ruff, mypy over 61 source files, four import contracts, lock validation, build, and diff checks.

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created.
- Added one locked benchmark command with deterministic synthetic profiles, explicit real-corpus input, raw JSON evidence, isolated spawn workers, equivalent-read checks, and recovery observations.
- Revalidated flat binary against Arrow IPC and Zarr v3 on two synthetic profiles and a 2,000,000-row FORGE subset; claims remain scoped to recorded local conditions.
- Removed the one-item runtime backend registry and selector while preserving lazy per-process memory maps and store validation.
- Pre-review gate: 679 tests passed and 3 live tests were intentionally deselected; all static, architecture, lock, build, and diff checks passed.

### File List

- `_bmad-output/implementation-artifacts/2-1-select-the-primary-store-through-a-reproducible-benchmark.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
- `.github/workflows/ci.yml`
- `README.md`
- `benchmarks/__init__.py`
- `benchmarks/storage/README.md`
- `benchmarks/storage/__init__.py`
- `benchmarks/storage/__main__.py`
- `benchmarks/storage/benchmark.py`
- `benchmarks/storage/candidates.py`
- `benchmarks/storage/results/2026-09-21-forge-kaggle-defog.json`
- `benchmarks/storage/results/2026-09-21-synthetic.json`
- `docs/adr/0005-bakeoff.py` (deleted)
- `docs/adr/0005-canonical-store-is-flat-binary.md`
- `docs/adr/0005-forge-conversion-proof.py` (deleted)
- `docs/adr/0005-worker-scaling.py` (deleted)
- `pyproject.toml`
- `src/dsio/data/readers.py`
- `src/dsio/data/store.py`
- `tests/benchmarking/test_storage_benchmark.py`
- `uv.lock`

### Change Log

- 2026-09-21: Created Story 2.1 and marked it ready for development.
- 2026-09-21: Implemented the reproducible benchmark and selected-store simplification; moved the story to review after the complete gate passed.
