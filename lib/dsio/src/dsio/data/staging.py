"""Staging: build a derived artifact once, keyed by the config that produced it.

The predecessor was a pluggable policy-and-codec framework. The job it did is
small: hash the config, name a path from it, build if the path is missing. A
project that later needs environment-sensitive keys adds a field to the dict it
passes in, which is ten lines rather than a Protocol.

A build that raises leaves nothing behind. A half-written stage that looks
complete is worse than a missing one, because the next run reuses it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from dsio.contracts.hashing import canonical_json


class StagingError(RuntimeError):
    """Raised when a stage could not be built."""


def stage(
    name: str,
    config: dict[str, Any],
    build: Callable[[Path], None],
    *,
    root: Path | None = None,
) -> Path:
    """Return the path to a staged artifact, building it if it is not there."""
    base = Path(root) if root is not None else Path("stores")
    key = sha256(canonical_json(config).encode()).hexdigest()[:16]
    target = base / name / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    partial = target.with_suffix(".partial")
    try:
        build(partial)
    except Exception as exc:
        _cleanup_partial(partial)
        raise StagingError(f"stage {name!r} failed to build: {exc}") from exc
    partial.replace(target)
    return target


def _cleanup_partial(partial: Path) -> None:
    """Remove a failed build's leftovers, whether ``build`` left a file or a directory.

    Never raises: a build can fail after creating a directory (e.g. a ``SignalStore``
    root, the default staging shape), and ``Path.unlink`` on a directory raises
    ``IsADirectoryError``. Left unguarded, that error would replace the real build
    error rather than let it propagate, and the half-built directory would survive —
    unusable forever, since a later retry sees ``target`` still missing, rebuilds into
    the same ``partial`` path, and the builder's own ``mkdir`` then hits
    ``FileExistsError``. Cleanup failure must never mask the original build error.
    """
    try:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial, ignore_errors=True)
        else:
            partial.unlink(missing_ok=True)
    except OSError:
        pass
