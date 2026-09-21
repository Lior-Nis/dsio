"""Versioned metadata and file names for the canonical store layout."""

from __future__ import annotations

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

    schema_version: int
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
