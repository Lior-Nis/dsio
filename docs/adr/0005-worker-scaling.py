"""Does the store ranking survive DataLoader workers?

The single-process bake-off measured one reader. Training uses N worker processes reading
concurrently, and the two candidates have different bottlenecks: Zarr's cost is CPU-side
Python work, which parallelises across processes; memmap is page-cache memcpy, which
contends on memory bandwidth. So the ranking is not obviously stable under concurrency.

Uses multiprocessing with the fork start method, which is exactly what a PyTorch DataLoader
does on Linux by default. Handles are opened lazily *inside* each worker, never inherited —
that is the correctness half of the question, and getting it wrong is a real footgun rather
than a hypothetical one.

Also reports CPU seconds per million windows. That number decides more than throughput:
a reader that burns a full core per worker is competing with augmentation and collation for
the cores you wanted to spend on them.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "stores_mp"
T = 40_000_000
W = 500
PER_WORKER = 12000
SEED = 7

_handle: dict = {}


def _mm(path: str):
    """Open a per-process memmap, cached per worker.

    Opening lazily in the worker is not an optimisation. A np.memmap created in the parent
    and passed to a spawn-based worker is pickled *by value* — it would serialise the whole
    array. FORGE hit the same class of bug and solved it the same way, with a per-worker
    lazily-opened handle.
    """
    key = ("mm", path, os.getpid())
    if key not in _handle:
        _handle[key] = np.memmap(path, dtype=np.float32, mode="r", shape=(T, 3))
    return _handle[key]


def _zarr(path: str):
    key = ("zarr", path, os.getpid())
    if key not in _handle:
        import zarr

        _handle[key] = zarr.open_array(path, mode="r")
    return _handle[key]


def _arrow(path: str):
    key = ("arrow", path, os.getpid())
    if key not in _handle:
        import pyarrow as pa

        src = pa.memory_map(path, "r")
        _handle[key] = pa.ipc.open_file(src).read_all()["v"].combine_chunks()
    return _handle[key]


def work(args) -> tuple[float, float]:
    kind, path, seed = args
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - W - 1, size=PER_WORKER)

    if kind == "memmap":
        h = _mm(path)
        read = lambda s: np.array(h[s : s + W])  # noqa: E731
    elif kind == "zarr":
        h = _zarr(path)
        read = lambda s: h[s : s + W]  # noqa: E731
    else:
        h = _arrow(path)
        read = lambda s: h[s : s + W].flatten().to_numpy(zero_copy_only=False)  # noqa: E731

    for s in starts[:200]:
        read(int(s))

    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    total = 0.0
    for s in starts:
        total += float(read(int(s)).sum())
    wall = time.perf_counter() - t0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    assert total == total
    return wall, cpu


def run(kind: str, path: str, workers: int) -> tuple[float, float]:
    """Aggregate throughput, excluding pool startup.

    Timing the outer wall clock would fold fork, imports and warm-up into the result, and
    that fixed cost amortises differently at 1 worker than at 8 — which manufactures
    superlinear scaling out of nothing. Each worker times only its own read loop; the
    slowest of those is the real elapsed time for the batch of work.
    """
    ctx = mp.get_context("fork")
    jobs = [(kind, path, SEED + i) for i in range(workers)]
    with ctx.Pool(workers) as pool:
        out = pool.map(work, jobs)
    elapsed = max(w for w, _ in out)
    cpu = sum(c for _, c in out)
    return (workers * PER_WORKER) / elapsed, cpu / (workers * PER_WORKER) * 1e6


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(0)
    print(f"building {T * 3 * 4 / 1e9:.2f} GB of signal...", flush=True)
    sig = rng.standard_normal((T, 3)).astype(np.float32)

    raw = OUT / "f.bin"
    sig.tofile(raw)

    import zarr

    zp = OUT / "z.zarr"
    z = zarr.create_array(store=str(zp), shape=(T, 3), chunks=(2048, 3), dtype="float32")
    for i in range(0, T, 1 << 21):
        z[i : i + (1 << 21)] = sig[i : i + (1 << 21)]

    import pyarrow as pa

    ap = OUT / "a.arrow"
    tbl = pa.table({"v": pa.FixedSizeListArray.from_arrays(pa.array(sig.reshape(-1)), 3)})
    with pa.OSFile(str(ap), "wb") as sink, pa.ipc.new_file(sink, tbl.schema) as w:
        w.write_table(tbl)
    del sig, tbl

    cores = os.cpu_count() or 1
    print(f"\n{cores} cores. aggregate windows/s, and CPU-seconds per 1M windows\n")
    header = "format      " + "".join(f"{n:>12d}w" for n in (1, 2, 4, 8))
    print(header)
    for kind, path in (("memmap", str(raw)), ("arrow", str(ap)), ("zarr", str(zp))):
        cells, cpus = [], []
        for n in (1, 2, 4, 8):
            tp, cpu = run(kind, path, n)
            cells.append(f"{tp:>12,.0f} ")
            cpus.append(cpu)
        print(f"{kind:<12s}" + "".join(cells))
        print(f"{'  cpu s/1M':<12s}" + "".join(f"{c:>12.1f} " for c in cpus))
