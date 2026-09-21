# 5. The canonical signal store is flat binary, memory-mapped

Status: accepted (revalidated 2026-09-21)

## Context

DSio needs one physical format for immutable, fixed-dtype numeric arrays read as contiguous
windows. Choosing several backends would move format decisions into every consumer without
evidence that the flexibility is useful. The choice therefore comes from a focused,
repeatable local-filesystem benchmark.

This decision covers the measured DSio workload: C-contiguous `float32` arrays shaped
`[rows, channels]`, local storage, immutable publication, and materialized contiguous
windows. It is not a claim about object storage, sparse/tabular data, mixed dtypes, remote
streaming, or N-dimensional sub-volume access.

## Reproducible benchmark

From the repository root, the complete synthetic benchmark is one locked command:

```bash
uv run --locked --group benchmark python -m benchmarks.storage \
  --output benchmarks/storage/results/local-synthetic.json
```

The checked-in run uses two deterministic 2,000,000-row, three-channel profiles, 500-row
windows, 2,000 ordered reads, 2,000 seeded random reads, and spawn-based worker counts of
one, two, and four. It compares:

- flat C-contiguous binary through `numpy.memmap`;
- Arrow IPC through PyArrow's memory-mapped file reader; and
- Zarr v3 with 2,048-row chunks and its default compressor.

Every read is materialized and hashed. A run fails if candidates produce different bytes.
Each candidate records build time and throughput, ordered/random/multi-worker read timings,
process peak RSS, disk bytes, file count, package versions, source digest, seed, machine,
Git state, and the observed behavior of a deliberately incomplete write. The JSON contains
raw worker measurements rather than only summaries.

The benchmark does not drop the operating-system page cache. A candidate is built before
it is read, so these are warm-cache-oriented local results. Timing rank is intentionally not
asserted in CI; tests assert workload determinism, candidate equivalence, report completeness,
and recovery classification.

## 2026-09-21 results

Environment: Linux 6.17 x86-64, Python 3.12, NumPy 2.5.2, PyArrow 25.0.1,
Zarr 3.4.0, and Numcodecs 0.17.0.
Exact environment strings and raw measurements are in:

- `benchmarks/storage/results/2026-09-21-synthetic.json`
- `benchmarks/storage/results/2026-09-21-forge-kaggle-defog.json`

### Synthetic profiles

| Workload | Candidate | Build MB/s | Ordered MB/s | Random windows/s | 1/2/4-worker windows/s | Peak RSS MB | Disk MB / files |
|---|---|---:|---:|---:|---:|---:|---:|
| smooth signal | flat binary | 4,065 | 1,405 | 206,713 | 211,852 / 379,251 / 829,430 | 119 | 24.00 / 1 |
| smooth signal | Arrow IPC | 204 | 119 | 20,637 | 20,004 / 34,483 / 67,553 | 156 | 24.00 / 1 |
| smooth signal | Zarr v3 | 62 | 17 | 2,934 | 2,748 / 5,380 / 10,854 | 159 | 21.17 / 978 |
| fixed items | flat binary | 2,819 | 1,407 | 104,119 | 217,713 / 448,132 / 881,462 | 197 | 24.00 / 1 |
| fixed items | Arrow IPC | 185 | 99 | 16,910 | 19,020 / 34,508 / 65,517 | 197 | 24.00 / 1 |
| fixed items | Zarr v3 | 57 | 14 | 2,101 | 2,624 / 5,803 / 8,937 | 197 | 13.77 / 978 |

### Representative FORGE corpus

The real-corpus run used the first 2,000,000 flattened rows of FORGE's project-owned
`kaggle_defog` accelerometer `accs` array, originally shaped `[230555, 500, 3]`. This retains
real values and compression behavior but is a subset of materialized windows, not a claim
about every FORGE session or the full 1.38 GB logical array.

| Candidate | Build MB/s | Ordered MB/s | Random windows/s | 1/2/4-worker windows/s | Peak RSS MB | Disk MB / files |
|---|---:|---:|---:|---:|---:|---:|
| flat binary | 4,731 | 1,410 | 104,554 | 207,534 / 415,690 / 841,190 | 89 | 24.00 / 1 |
| Arrow IPC | 192 | 120 | 19,866 | 20,631 / 37,676 / 69,577 | 157 | 24.00 / 1 |
| Zarr v3 | 62 | 17 | 2,715 | 2,618 / 5,649 / 11,447 | 159 | 9.05 / 978 |

The real workload can be rerun where that project-owned corpus is available:

```bash
uv run --locked --group benchmark python -m benchmarks.storage \
  --real-only --real-zarr /path/to/corpus.zarr --real-key accs \
  --real-name forge-kaggle-defog --real-row-limit 2000000 \
  --output benchmarks/storage/results/local-forge.json
```

## Recovery and operational observations

Flat binary and Arrow each produced one payload file; Zarr produced 978 files for each
2,000,000-row workload. Flat binary needs only NumPy, Arrow adds PyArrow, and Zarr adds Zarr
and its codec/storage dependencies. These are observable facts, not a subjective complexity
score.

An incomplete flat payload could not be opened with its declared full shape. Incomplete
Arrow and Zarr outputs could be opened, but neither matched the expected shape-and-content
digest. All candidates rebuilt successfully after the partial output was discarded. Thus
no candidate removes the need for DSio's atomic publication, manifest, and digest checks.

## Historical evidence

The 2026-08 exploratory run used the full 1.38 GB logical FORGE array and reported 317,866
random windows/s for flat binary, 247,942 for Arrow IPC, 2,336–4,937 for Zarr variants, and
4,268 for Lance. A separate fork-worker run reported 405,034 / 2,866,337 windows/s for flat
binary at one/eight workers, 262,615 / 1,901,997 for Arrow, and 3,054 / 26,104 for Zarr.

Those results remain useful corroboration but came from hard-coded, undeclared scripts and
are not treated as the reproducible record. The scripts were removed after their relevant
candidate coverage and lessons were incorporated into `benchmarks.storage`.

## Decision

The one canonical DSio payload is flat C-contiguous binary with a versioned DSio index and
manifest, read locally through `numpy.memmap`.

Across the declared runs it had the highest ordered, random, and multi-worker throughput,
the smallest dependency surface, and one payload file. Zarr materially reduced disk use on
compressible data, but that trade cost substantial build/read throughput and hundreds of
files. Arrow preserves a richer schema that DSio's homogeneous numeric payload does not
need and was slower in these measured access paths.

DSio therefore exposes no backend selector or runtime storage registry. A future workload
that invalidates these conditions must first extend and rerun this benchmark; a changed
physical implementation remains behind the stable store boundary rather than becoming a
project-side branch.

## Consequences

- Readers remain lazy and per-process so spawn workers reopen mappings instead of pickling
  payload bytes.
- The store remains immutable after atomic publication and is rejected when index,
  manifest, shape, or digest checks fail.
- Compression, remote/object storage, mixed schemas, and arbitrary N-dimensional chunks
  are not silently promised by this decision.
- Performance claims in DSio documentation must cite the measured workload and result file.
