# 5. The canonical signal store is flat binary, memory-mapped

Status: accepted (2026-08-15)
Supersedes: the "canonical store format" open question in the design spec.

## Context

The design spec left one question open: what backs a canonical store of continuous signal
`[T, C]` from which windows are read at arbitrary offsets? That is the SSL access pattern —
masked and contrastive pretraining sample random offsets thousands of times per second.

The spec's provisional default was Zarr v3, on the grounds that FORGE proved it at 229 GB.
The stated hypothesis was that **Lance** might win, because it claims roughly 100× faster
random access than Parquet *and* carries zero-copy dataset versioning in the format itself,
which would collapse the store and the manifest into one thing.

Both were tested against real data rather than argued about.

## Method

Real FORGE accelerometer windows (`len500_stride200_fogstride100_anyfog_kaggle_defog.zarr`,
230,555 × 500 × 3 float32) flattened back into one continuous signal of 115,277,500 × 3 —
1.38 GB. Each candidate was written, warmed, and then timed on 2,000 random 500-step window
reads. Every read is forced to materialise via `.sum()`: a numpy memmap slice is a *view*,
and timing it without touching the bytes measures view construction and reports a number
that is pure fiction. An earlier run made exactly that mistake and reported 1.3M win/s.

## Results

| Format | Random windows/s | Relative | Size |
|---|---|---|---|
| **np.memmap flat binary** | **317,866** | 1.00× | 1.38 GB |
| Arrow IPC, memory-mapped | 247,942 | 0.78× | 1.38 GB |
| Zarr v3, chunk 8192, uncompressed | 4,937 | 0.016× | 1.38 GB |
| Lance | 4,268 | 0.013× | 1.39 GB |
| Zarr v3, chunk 8192, blosc/lz4 | 3,627 | 0.011× | 0.44 GB |
| Zarr v3, chunk 65536, blosc/lz4 | 2,336 | 0.007× | 0.41 GB |

A second run gave Zarr its best case — chunks sized to the window, so read amplification is
~1× instead of 16× — on 20M synthetic timesteps:

| Format | Random windows/s |
|---|---|
| np.memmap | 390,634 |
| Zarr chunk 2048, uncompressed | 4,728 |
| Zarr chunk 1024, uncompressed | 4,144 |
| Zarr chunk 512, uncompressed | 3,393 |
| Zarr chunk 512, blosc/lz4 | 2,361 |

## Findings

**Memory-mapped formats are roughly 65–90× faster than chunked ones for this access
pattern.** That is not a margin you tune away.

**The cost is not decompression.** Uncompressed Zarr is only 1.4× faster than compressed
Zarr, and chunk tuning moves it by less than 2×. The bottleneck is the chunked-array access
path itself — per-call indexing machinery in Python — so neither a faster codec nor a better
chunk shape rescues it.

**The Lance hypothesis was wrong**, and it is worth being precise about why rather than
just recording the number. Lance's `take()` gathers *rows* by index. Asking for 500
consecutive rows is a 500-element row gather, not a contiguous slice, so its strength —
random access across wide or multimodal records — is the wrong strength for narrow numeric
signal read in runs. It is 74× slower than memmap here. Its built-in versioning was the real
attraction, and that is simply not worth 74×.

**Compression buys 3.1×space** (1.38 GB → 0.44 GB). Real, but not the binding constraint:
FORGE's 229 GB was mostly duplicated materialised windows, which the lazy-view design
eliminates regardless of codec.

## Decision

The canonical continuous-signal store is a **flat binary payload plus a versioned index**,
memory-mapped for local reads — Megatron-LM's `.bin`/`.idx` and nanoGPT's `train.bin`,
arrived at by measurement rather than by imitation.

**Arrow IPC memory-mapped is the sanctioned alternative** where a schema, named columns, or
mixed dtypes are needed. At 0.78× it costs little, and it is the substrate HuggingFace
`datasets` already uses.

Zarr and Lance are **not** the canonical store. Zarr keeps a narrow role for genuinely
N-dimensional data sampled as sub-volumes, where its chunking earns its cost; nothing in the
v1 modalities qualifies. Versioning stays in the manifest layer, where ADR 0002 already put
it.

## Consequences

The `StorageBackend` ABC from ADR 0004 is now load-bearing rather than tidy. These numbers
are **warm page cache**, which is the honest workstation case and the optimistic cloud case
— you cannot mmap S3. The remote backend will use ranged reads against the same index
semantics, exactly as Megatron's `_BinReader` has mmap, plain-file, and S3 implementations
behind one `.idx`. Local and remote will not have comparable throughput, and the design must
not pretend otherwise: the answer for cloud is a node-local NVMe cache, not a faster reader.

Compression becomes a per-store option rather than a default, off for anything in the
training hot path.

## Caveats

Warm cache; `drop_caches` needs root. Narrow signal (3 channels) — Lance may well win on
wide or multimodal records, and this result should not be generalised to those. Single
process, no dataloader workers. The benchmark lives at
`scratchpad/bakeoff.py` and should be re-run if the access pattern changes.
