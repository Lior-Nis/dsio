"""Storage backends for a signal payload.

The index semantics are identical regardless of where the bytes live; only the fetch
differs. That separation is what lets the Zarr-versus-flat-binary decision (ADR 0005) stay
inside this module, and it is the shape Megatron-LM uses — one ``.idx``, several
``_BinReader`` implementations.

Every reader is opened **per process**. This is not an optimisation. A ``np.memmap``
created in a parent process and handed to a ``spawn``-based DataLoader worker is pickled
*by value*: it serialises the whole array instead of the mapping. Linux forks and inherits
the mapping harmlessly, so the bug stays hidden until someone runs on macOS, or sets
``multiprocessing_context="spawn"`` to dodge a CUDA fork issue. FORGE hit the same class of
bug with Zarr handles and fixed it the same way.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class ReadError(RuntimeError):
    """Raised when a read cannot be satisfied."""


@runtime_checkable
class SignalReader(Protocol):
    """Fetch a contiguous run of rows from a signal payload."""

    def read_rows(self, start: int, n_rows: int) -> np.ndarray: ...

    def close(self) -> None: ...


class MmapReader:
    """Memory-mapped local reads. The default, and by a wide margin the fastest.

    Benchmarked at ~318k random 500-step windows/s single-process and ~2.9M at eight
    workers, using 2.7 CPU-seconds per million windows — roughly 120x less CPU than a
    chunked format. See ADR 0005.
    """

    def __init__(self, path: Path, dtype: np.dtype, channels: int, n_rows: int) -> None:
        self._array = np.memmap(path, dtype=dtype, mode="r", shape=(n_rows, channels))

    def read_rows(self, start: int, n_rows: int) -> np.ndarray:
        # np.array, not np.asarray: a memmap slice is a view, and handing a view to a
        # consumer keeps the mapping alive and hides the true cost of the read.
        return np.array(self._array[start : start + n_rows])

    def close(self) -> None:
        self._array = None  # type: ignore[assignment]


class FileReader:
    """Positional reads through the file API, for filesystems where mmap misbehaves.

    Network filesystems are the usual reason: an mmap over NFS can turn a page fault into
    a silent stall or a SIGBUS on truncation, which is far worse than a slow read.
    """

    def __init__(self, path: Path, dtype: np.dtype, channels: int, n_rows: int) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        self._dtype = dtype
        self._channels = channels
        self._n_rows = n_rows
        self._row_bytes = int(dtype.itemsize) * channels

    def read_rows(self, start: int, n_rows: int) -> np.ndarray:
        want = n_rows * self._row_bytes
        raw = os.pread(self._fd, want, start * self._row_bytes)
        if len(raw) != want:
            raise ReadError(
                f"short read at row {start}: wanted {want} bytes, got {len(raw)}; "
                "the payload is truncated"
            )
        return np.frombuffer(raw, dtype=self._dtype).reshape(n_rows, self._channels)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


BACKENDS: dict[str, type] = {"mmap": MmapReader, "file": FileReader}


def open_reader(
    backend: str, path: Path, dtype: np.dtype, channels: int, n_rows: int
) -> SignalReader:
    """Open a reader by backend name, failing on an unknown one rather than defaulting.

    Silently falling back to mmap when a caller asked for something else would hide the
    very condition they were working around.
    """
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ReadError(
            f"unknown storage backend {backend!r}; known: {', '.join(sorted(BACKENDS))}"
        ) from None
    return cls(path, dtype, channels, n_rows)
