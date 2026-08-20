"""Storage backends for a signal payload.

The index semantics are identical regardless of where the bytes live; only the fetch
differs. That separation is what lets the Zarr-versus-flat-binary decision (ADR 0005) stay
inside this module, and it is the shape Megatron-LM uses — one ``.idx``, several
``_BinReader`` implementations.

Every reader is opened **per process**. This is not an optimisation. A ``np.memmap``
created in a parent process and handed to a ``spawn``-based DataLoader worker is pickled
*by value*: it serialises the whole array instead of the mapping. Linux forks and inherits
the mapping harmlessly, so the bug stays hidden until someone runs on macOS, or sets
``multiprocessing_context="spawn"`` to dodge a CUDA fork issue. This is the same class of
bug with Zarr handles and fixed it the same way.
"""

from __future__ import annotations

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


BACKENDS: dict[str, type] = {"mmap": MmapReader}


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
            f"unknown backend {backend!r}; the store is memory-mapped (ADR 0005). "
            "A sharded streaming backend is the planned extension, not a fallback."
        ) from None
    return cls(path, dtype, channels, n_rows)
