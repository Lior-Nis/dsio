"""Canonical-store bake-off on real FORGE accelerometer data.

The question: what backs a canonical store of continuous signal [T, C] from which we read
random windows? That is the SSL access pattern — masked or contrastive pretraining samples
arbitrary offsets, thousands of times per second.

Method: take real FORGE windows, flatten them back into one continuous [T, 3] float32
signal, write it in each candidate format, then measure random-window reads.

Caveat recorded up front: the page cache cannot be dropped without root, so these are
warm-cache numbers. That is the realistic case for a workstation and the optimistic case
for cloud storage, and it biases toward formats that would otherwise pay decode cost.
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import numpy as np

SRC = "/home/liornisimov/Projects/acc_base/data/processed/len500_stride200_fogstride100_anyfog_kaggle_defog.zarr"
OUT = Path("/tmp/claude-1000/-home-liornisimov-Projects-dsio/7c1160a1-b184-4045-87ce-012608c7cc9f/scratchpad/stores")
WINDOW = 500
N_READS = 2000
BATCH = 64
SEED = 42


def dir_size(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def load_signal() -> np.ndarray:
    import zarr

    z = zarr.open(SRC, mode="r")
    accs = z["accs"]
    n = accs.shape[0]
    print(f"reading {n} windows from FORGE ({accs.nbytes / 1e9:.2f} GB)...", flush=True)
    block = accs[:]
    signal = np.ascontiguousarray(block.reshape(-1, 3))
    del block
    gc.collect()
    print(f"continuous signal: {signal.shape} = {signal.nbytes / 1e9:.2f} GB", flush=True)
    return signal


def bench(name: str, read_window, read_batch, size_gb: float, results: list) -> None:
    """Time random-window reads.

    Two things this has to get right. Every read is forced to materialise via .sum(),
    because a numpy memmap slice is a *view* — timing it without touching the bytes
    measures view construction and reports a number that is pure fiction. And each format
    gets a warm-up pass, because the first loop over a fresh mapping pays every page
    fault and would penalise whichever format is measured first.
    """
    rng = np.random.default_rng(SEED)
    max_start = TOTAL - WINDOW - 1

    for s in rng.integers(0, max_start, size=200):
        read_window(int(s))
    if read_batch is not None:
        read_batch([int(s) for s in rng.integers(0, max_start, size=BATCH)])

    total = 0.0
    starts = rng.integers(0, max_start, size=N_READS)
    t0 = time.perf_counter()
    for s in starts:
        total += float(read_window(int(s)).sum())
    single = N_READS / (time.perf_counter() - t0)

    batched = float("nan")
    if read_batch is not None:
        n_batches = max(1, N_READS // BATCH)
        starts = rng.integers(0, max_start, size=(n_batches, BATCH))
        t0 = time.perf_counter()
        for row in starts:
            total += float(read_batch([int(s) for s in row]).sum())
        batched = (n_batches * BATCH) / (time.perf_counter() - t0)

    assert total == total  # NaN would mean a format silently returned garbage
    results.append((name, single, batched, size_gb))
    print(f"  {name:28s} {single:9.0f} win/s   {batched:9.0f} win/s (b{BATCH})   {size_gb:5.2f} GB",
          flush=True)


if __name__ == "__main__":
    signal = load_signal()
    TOTAL = signal.shape[0]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    results: list = []

    print(f"\nrandom {WINDOW}-step window reads, {N_READS} single + batched at {BATCH}\n")

    # --- np.memmap over flat binary: the nanoGPT / Megatron .bin shape -------------
    raw = OUT / "flat.bin"
    signal.tofile(raw)
    mm = np.memmap(raw, dtype=np.float32, mode="r", shape=(TOTAL, 3))
    bench(
        "np.memmap flat binary",
        lambda s: np.array(mm[s : s + WINDOW]),
        lambda ss: np.stack([mm[s : s + WINDOW] for s in ss]),
        dir_size(raw),
        results,
    )

    # --- Zarr v3, two chunk sizes ---------------------------------------------------
    import zarr

    for chunk_rows, compressed in ((8192, True), (65536, True), (8192, False)):
        tag = f"c{chunk_rows}" + ("" if compressed else " raw")
        path = OUT / f"zarr_c{chunk_rows}_{'z' if compressed else 'raw'}.zarr"
        z = zarr.create_array(
            store=str(path),
            shape=(TOTAL, 3),
            chunks=(chunk_rows, 3),
            dtype="float32",
            compressors=None if not compressed else "auto",
        )
        step = 1 << 21
        for i in range(0, TOTAL, step):
            z[i : i + step] = signal[i : i + step]
        zr = zarr.open_array(str(path), mode="r")
        bench(
            f"zarr v3 {tag}",
            lambda s, zr=zr: zr[s : s + WINDOW],
            lambda ss, zr=zr: np.stack([zr[s : s + WINDOW] for s in ss]),
            dir_size(path),
            results,
        )

    # --- Arrow IPC, memory-mapped ---------------------------------------------------
    import pyarrow as pa

    flat = pa.FixedSizeListArray.from_arrays(pa.array(signal.reshape(-1)), 3)
    table = pa.table({"v": flat})
    ipc = OUT / "signal.arrow"
    with pa.OSFile(str(ipc), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    src = pa.memory_map(str(ipc), "r")
    reader = pa.ipc.open_file(src)
    at = reader.read_all()["v"].combine_chunks()

    def arrow_window(s: int) -> np.ndarray:
        return at[s : s + WINDOW].flatten().to_numpy(zero_copy_only=False).reshape(-1, 3)

    bench(
        "arrow IPC mmap",
        arrow_window,
        lambda ss: np.stack([arrow_window(s) for s in ss]),
        dir_size(ipc),
        results,
    )

    # --- Lance ----------------------------------------------------------------------
    import lance

    lpath = OUT / "signal.lance"
    lance.write_dataset(table, str(lpath), mode="overwrite")
    ds = lance.dataset(str(lpath))

    def lance_window(s: int) -> np.ndarray:
        t = ds.take(list(range(s, s + WINDOW)), columns=["v"])
        return t["v"].combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, 3)

    def lance_batch(ss: list[int]) -> np.ndarray:
        idx: list[int] = []
        for s in ss:
            idx.extend(range(s, s + WINDOW))
        t = ds.take(idx, columns=["v"])
        arr = t["v"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
        return arr.reshape(len(ss), WINDOW, 3)

    bench("lance", lance_window, lance_batch, dir_size(lpath), results)

    print("\n--- summary (higher is better) ---")
    best = max(r[1] for r in results)
    for name, single, batched, gb in sorted(results, key=lambda r: -r[1]):
        print(f"{name:28s} {single:9.0f} win/s  {single / best:5.2f}x  {gb:5.2f} GB")
