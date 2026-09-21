"""Validation and construction of immutable MLflow evidence references."""

from __future__ import annotations

import re
from collections.abc import Collection
from pathlib import PurePosixPath

from dsio.tracking._lifecycle import TrackingError

_RUN_ID = re.compile(r"[0-9a-f]{32}")


def evidence_uri(run_id: str, artifact_path: str) -> str:
    """Build an immutable MLflow Runs artifact URI from exact identifiers."""
    require_run_id(run_id)
    path = validate_artifact_path(artifact_path)
    return f"runs:/{run_id}/{path}"


def artifact_paths(paths: Collection[str]) -> tuple[str, ...]:
    values = (paths,) if isinstance(paths, str) else tuple(paths)
    invalid = sorted({type(path).__name__ for path in values if not isinstance(path, str)})
    if invalid:
        raise TrackingError(f"Artifact paths must be str, got: {', '.join(invalid)}.")
    return tuple(dict.fromkeys(validate_artifact_path(path) for path in values))


def validate_artifact_path(path: str) -> str:
    if not isinstance(path, str):
        raise TrackingError(
            f"Evidence artifact path must be str, got {type(path).__name__}."
        )
    candidate = PurePosixPath(path)
    if (
        not path
        or path == "."
        or "\\" in path
        or "?" in path
        or "#" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or str(candidate) != path
    ):
        raise TrackingError(
            "Evidence requires a non-empty relative POSIX artifact path without traversal."
        )
    return path


def require_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise TrackingError(
            "Evidence requires an immutable 32-character MLflow Run ID; mutable model "
            "aliases and stages are not accepted."
        )
