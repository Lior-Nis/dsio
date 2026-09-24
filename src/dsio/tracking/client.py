"""Small shared settings for native MLflow client operations."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_TRACKING_URI = "http://localhost:5000"


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve an explicit URI, MLflow's environment variable, or the local default."""
    if tracking_uri is not None:
        return tracking_uri
    return os.environ.get(TRACKING_URI_ENV) or DEFAULT_TRACKING_URI


@contextmanager
def temporary_mlflow_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set process environment values for one MLflow operation, then restore them."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
