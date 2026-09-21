"""Versioned metadata and file names for the canonical store layout."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from pydantic import Field

from dsio.contracts import DsioModel

DATA_ROOT_ENV = "DSIO_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("stores")

SIGNAL_FILE = "signal.bin"
INDEX_FILE = "signal.idx"
ENTITIES_FILE = "entities.jsonl"
MANIFEST_FILE = "manifest.yaml"
STORE_SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    """Raised when a store is malformed, incomplete, or contradicts its manifest."""


class Entity(DsioModel):
    """One stable storage sample occupying contiguous rows in the payload."""

    entity_id: str
    group: str
    start_row: int
    n_rows: int
    data_sha256: str
    attrs: dict[str, Any] = Field(default_factory=dict)

    @property
    def end_row(self) -> int:
        return self.start_row + self.n_rows


class StoreManifest(DsioModel):
    """Committed description of one immutable store."""

    schema_version: int = Field(strict=True)
    name: str
    created_at: str
    dtype: str
    channels: int
    n_rows: int
    n_entities: int
    n_groups: int
    signal_sha256: str
    index_sha256: str
    entities_sha256: str
    signal_bytes: int
    source: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class StoredSample(TypedDict):
    """The stable public value returned for one persisted sample."""

    sample_id: str
    data: np.ndarray
    group: str
    attrs: dict[str, Any]


def validated_attrs(attrs: dict[str, Any] | None, field: str) -> dict[str, Any]:
    """Return an independent JSON-compatible copy or fail before publication."""
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        raise StoreError(f"{field} must be a JSON object")
    try:
        return _copy_json_object(attrs, field, set())
    except RecursionError:
        raise StoreError(f"{field} is nested too deeply to be valid JSON metadata") from None


def _copy_json_object(
    value: dict[Any, Any], field: str, active: set[int]
) -> dict[str, Any]:
    _enter_container(value, field, active)
    try:
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StoreError(f"{field} must use string JSON object keys, got {key!r}")
            copied[key] = _copy_json_value(item, f"{field}.{key}", active)
        return copied
    finally:
        active.remove(id(value))


def _copy_json_value(value: Any, field: str, active: set[int]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise StoreError(f"{field} must be a finite JSON number")
    if isinstance(value, list):
        _enter_container(value, field, active)
        try:
            return [
                _copy_json_value(item, f"{field}[{index}]", active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(id(value))
    if isinstance(value, dict):
        return _copy_json_object(value, field, active)
    raise StoreError(f"{field} contains non-JSON value {type(value).__name__}")


def _enter_container(value: object, field: str, active: set[int]) -> None:
    identity = id(value)
    if identity in active:
        raise StoreError(f"{field} contains a cycle and is not valid JSON metadata")
    active.add(identity)
