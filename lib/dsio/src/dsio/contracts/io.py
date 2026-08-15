"""Durable file writes.

Used by both the run ledger and the model registry. A partially-written run record or
model artifact is worse than a missing one, because it looks complete.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically and durably.

    Three steps matter and all three are load-bearing: write to a sibling temp file so a
    crash never leaves ``path`` half-written; fsync the file so the bytes reach the disk
    rather than the page cache; and fsync the *directory* so the rename itself is
    durable. Skipping the last one leaves a correctly-written file the filesystem has not
    yet linked — a record that exists in cache and nowhere else.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def fsync_dir(directory: Path) -> None:
    """Flush a directory entry so a rename or create survives a crash."""
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
