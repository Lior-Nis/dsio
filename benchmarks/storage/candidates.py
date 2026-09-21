"""Candidate-specific I/O for the storage benchmark.

This module is deliberately benchmark-only. DSio's production store has one physical
implementation and no backend registry.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]

CANDIDATES = ("flat-binary", "arrow-ipc", "zarr-v3")
RSS_METHOD = (
    "resource.getrusage(RUSAGE_SELF).ru_maxrss" if resource is not None else None
)


def build_candidate(candidate: str, source_path: Path, target: Path) -> dict[str, Any]:
    """Build one candidate and return the raw isolated-process measurement."""
    source = np.load(source_path, mmap_mode="r")
    started = time.perf_counter()
    if candidate == "flat-binary":
        source.tofile(target)
        configuration: dict[str, Any] = {"layout": "C-contiguous", "compression": None}
        dependencies = ["numpy"]
    elif candidate == "arrow-ipc":
        import pyarrow as pa

        values = pa.FixedSizeListArray.from_arrays(
            pa.array(np.asarray(source).reshape(-1)), source.shape[1]
        )
        table = pa.table({"values": values})
        with pa.OSFile(str(target), "wb") as sink, pa.ipc.new_file(
            sink, table.schema
        ) as writer:
            writer.write_table(table)
        configuration = {"container": "IPC file", "compression": None}
        dependencies = ["numpy", "pyarrow"]
    elif candidate == "zarr-v3":
        import zarr

        chunk_rows = min(2048, source.shape[0])
        codec = zarr.codecs.ZstdCodec(level=0, checksum=False)
        zarr.create_array(
            store=str(target),
            data=np.asarray(source),
            chunks=(chunk_rows, source.shape[1]),
            compressors=(codec,),
            zarr_format=3,
            overwrite=False,
        )
        configuration = {
            "format": 3,
            "chunks": [chunk_rows, int(source.shape[1])],
            "compressors": [codec.to_dict()],
        }
        dependencies = ["numpy", "zarr"]
    else:
        raise ValueError(f"unknown benchmark candidate {candidate!r}")

    elapsed = time.perf_counter() - started
    storage = storage_facts(target)
    return {
        "elapsed_seconds": elapsed,
        "throughput_bytes_per_second": source.nbytes / elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "configuration": configuration,
        "dependencies": dependencies,
        "storage": storage,
    }


def measure_reads(
    candidate: str,
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: str,
    starts: list[int],
    window: int,
) -> dict[str, Any]:
    """Materialize the requested windows and return raw timing and integrity evidence."""
    handle = _open(candidate, path, shape=shape, dtype=np.dtype(dtype))
    return _measure_open_handle(candidate, handle, shape, dtype, starts, window)


def measure_reads_synchronized(
    candidate: str,
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: str,
    starts: list[int],
    window: int,
    ready: Any,
    start: Any,
) -> dict[str, Any]:
    """Open first, prove worker readiness, then measure after a shared release."""
    handle = _open(candidate, path, shape=shape, dtype=np.dtype(dtype))
    pid = os.getpid()
    ready.put(pid)
    if not start.wait(timeout=60):
        raise RuntimeError("timed out waiting for the synchronized read start")
    result = _measure_open_handle(candidate, handle, shape, dtype, starts, window)
    result["pid"] = pid
    return result


def _measure_open_handle(
    candidate: str,
    handle: Any,
    shape: tuple[int, int],
    dtype: str,
    starts: list[int],
    window: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    for start in starts:
        rows = _read(candidate, handle, start, window)
        digest.update(np.ascontiguousarray(rows).tobytes())
    elapsed = time.perf_counter() - started
    logical_bytes = len(starts) * window * shape[1] * np.dtype(dtype).itemsize
    return {
        "elapsed_seconds": elapsed,
        "windows": len(starts),
        "logical_bytes": logical_bytes,
        "windows_per_second": len(starts) / elapsed,
        "throughput_bytes_per_second": logical_bytes / elapsed,
        "checksum": digest.hexdigest(),
        "peak_rss_bytes": peak_rss_bytes(),
    }


def recovery_observation(
    candidate: str,
    source: np.ndarray,
    root: Path,
) -> dict[str, Any]:
    """Observe whether a half-written candidate can be mistaken for complete output."""
    root.mkdir(parents=True, exist_ok=True)
    partial = root / "partial" / candidate_path(candidate)
    published = root / "published" / candidate_path(candidate)
    partial.parent.mkdir(parents=True, exist_ok=True)
    expected = hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest()
    source_path = root / "source.npy"
    np.save(source_path, source, allow_pickle=False)
    build_candidate(candidate, source_path, partial)
    _simulate_interruption(candidate, partial)

    opened, accepted = _classify_partial(
        candidate,
        partial,
        shape=source.shape,
        dtype=source.dtype,
        expected_digest=expected,
    )
    _remove(partial)

    published.parent.mkdir(parents=True, exist_ok=True)
    was_published = published.exists()
    build_candidate(candidate, source_path, published)
    _, retry_succeeded = _classify_partial(
        candidate,
        published,
        shape=source.shape,
        dtype=source.dtype,
        expected_digest=expected,
    )
    _remove(published)
    source_path.unlink()

    return {
        "rows": len(source),
        "published": was_published,
        "partial_opened": opened,
        "partial_accepted_as_complete": accepted,
        "retry_succeeded": retry_succeeded,
    }


def candidate_path(candidate: str) -> Path:
    if candidate == "flat-binary":
        return Path("data.bin")
    if candidate == "arrow-ipc":
        return Path("data.arrow")
    if candidate == "zarr-v3":
        return Path("data.zarr")
    raise ValueError(f"unknown benchmark candidate {candidate!r}")


def storage_facts(path: Path) -> dict[str, int]:
    if path.is_file():
        return {"bytes": path.stat().st_size, "files": 1}
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"bytes": sum(item.stat().st_size for item in files), "files": len(files)}


def peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _simulate_interruption(candidate: str, path: Path) -> None:
    """Model an interrupted publication by removing bytes from a real built output."""
    if candidate in {"flat-binary", "arrow-ipc"}:
        with path.open("r+b") as payload:
            payload.truncate(max(1, path.stat().st_size // 2))
        return
    if candidate == "zarr-v3":
        chunks = sorted(
            item for item in path.rglob("*") if item.is_file() and item.name != "zarr.json"
        )
        if not chunks:
            (path / "zarr.json").unlink()
            return
        chunks[-1].unlink()
        return
    raise ValueError(f"unknown benchmark candidate {candidate!r}")


def _open(
    candidate: str,
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: np.dtype[Any],
) -> Any:
    if candidate == "flat-binary":
        return np.memmap(path, dtype=dtype, mode="r", shape=shape)
    if candidate == "arrow-ipc":
        import pyarrow as pa

        source = pa.memory_map(str(path), "r")
        values = pa.ipc.open_file(source).read_all()["values"].combine_chunks()
        return source, values
    if candidate == "zarr-v3":
        import zarr

        return zarr.open_array(str(path), mode="r")
    raise ValueError(f"unknown benchmark candidate {candidate!r}")


def _read(candidate: str, handle: Any, start: int, rows: int) -> np.ndarray:
    if candidate == "flat-binary":
        return np.array(handle[start : start + rows])
    if candidate == "arrow-ipc":
        _, values = handle
        width = values.type.list_size
        flattened = values.values.slice((values.offset + start) * width, rows * width)
        return flattened.to_numpy(zero_copy_only=False).reshape(-1, width)
    if candidate == "zarr-v3":
        return np.asarray(handle[start : start + rows])
    raise ValueError(f"unknown benchmark candidate {candidate!r}")


def _classify_partial(
    candidate: str,
    path: Path,
    *,
    shape: tuple[int, int],
    dtype: np.dtype[Any],
    expected_digest: str,
) -> tuple[bool, bool]:
    try:
        handle = _open(candidate, path, shape=shape, dtype=dtype)
        rows = _read(candidate, handle, 0, shape[0])
    except (OSError, ValueError, EOFError):
        return False, False
    digest = hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()
    return True, rows.shape == shape and digest == expected_digest


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
