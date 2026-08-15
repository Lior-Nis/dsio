"""Sinks that mirror the run ledger into an external tracker.

Sinks are strictly secondary. The ledger on disk is authoritative; MLflow and W&B are
views onto it. That ordering is what lets a run happen on a plane, inside a Kaggle
kernel, or after a vendor changes its schema.

Because a sink is never the source of truth, a sink failure must never kill a run — but
it must also never pass silently, or you discover at the end of a week that nothing was
mirrored. Failures are logged loudly and the run continues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ExperimentTracker(Protocol):
    """The interface every tracking sink implements."""

    def start_run(self, run_id: str, name: str, tags: dict[str, str]) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def finish(self, status: str) -> None: ...


class NullTracker:
    """The default sink. Does nothing, successfully."""

    def start_run(self, run_id: str, name: str, tags: dict[str, str]) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def finish(self, status: str) -> None: ...


class MultiTracker:
    """Fans calls out to several sinks, isolating each one's failures.

    A broken sink degrades to a warning rather than taking the training run with it. The
    warning names the sink, because a silent tracking gap is its own kind of data loss.
    """

    def __init__(self, sinks: list[ExperimentTracker]) -> None:
        self._sinks = list(sinks)

    def _each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for sink in self._sinks:
            try:
                getattr(sink, method)(*args, **kwargs)
            except Exception:
                logger.warning(
                    "tracking sink %s failed during %s; the run continues and the "
                    "ledger on disk remains authoritative",
                    type(sink).__name__,
                    method,
                    exc_info=True,
                )

    def start_run(self, run_id: str, name: str, tags: dict[str, str]) -> None:
        self._each("start_run", run_id, name, tags)

    def log_params(self, params: dict[str, Any]) -> None:
        self._each("log_params", params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._each("log_metrics", metrics, step)

    def log_artifact(self, path: Path) -> None:
        self._each("log_artifact", path)

    def finish(self, status: str) -> None:
        self._each("finish", status)


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into dotted keys for trackers that only accept scalars."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(value, path))
        elif isinstance(value, list | tuple):
            flat[path] = ",".join(str(item) for item in value)
        else:
            flat[path] = value
    return flat
