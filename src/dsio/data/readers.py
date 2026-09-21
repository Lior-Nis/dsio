"""Memory-mapped reads for a signal payload.

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


@runtime_checkable
class SignalReader(Protocol):
    """Fetch a contiguous run of rows from a signal payload."""

    def read_rows(self, start: int, n_rows: int) -> np.ndarray: ...

    def close(self) -> None: ...


class MmapReader:
    """Memory-mapped local reads, selected by the scoped benchmark in ADR 0005."""

    def __init__(self, path: Path, dtype: np.dtype, channels: int, n_rows: int) -> None:
        self._array = np.memmap(path, dtype=dtype, mode="r", shape=(n_rows, channels))

    def read_rows(self, start: int, n_rows: int) -> np.ndarray:
        # np.array, not np.asarray: a memmap slice is a view, and handing a view to a
        # consumer keeps the mapping alive and hides the true cost of the read.
        return np.array(self._array[start : start + n_rows])

    def close(self) -> None:
        self._array = None  # type: ignore[assignment]


def open_reader(path: Path, dtype: np.dtype, channels: int, n_rows: int) -> SignalReader:
    """Open the benchmark-selected flat-binary reader."""
    return MmapReader(path, dtype, channels, n_rows)
