"""Prefect-native cache keys for pure deterministic task values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dsio.tracking._lifecycle import TrackingError
from dsio.tracking.provenance import execution_identity


def prefect_cache_key(context: Any, parameters: Mapping[str, Any]) -> str:
    """Derive a Prefect cache key from pure serializable task parameters."""
    task = getattr(context, "task", None)
    task_key = getattr(task, "task_key", None)
    if not isinstance(task_key, str) or not task_key:
        raise TrackingError("A Prefect task context with a stable task key is required.")
    return execution_identity({"task_key": task_key, "parameters": parameters})
