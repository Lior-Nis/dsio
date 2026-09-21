"""Append-only construction of the canonical flat-binary store."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import yaml

from dsio.contracts import atomic_write, sha256_of_bytes, sha256_of_file
from dsio.data.format import FORMAT_VERSION, DTypeCode, IndexFormatError, IndexHeader, write_index
from dsio.data.store.layout import (
    ENTITIES_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    SIGNAL_FILE,
    STORE_SCHEMA_VERSION,
    Entity,
    StoreError,
    StoreManifest,
    validated_attrs,
)


class SignalStoreBuilder:
    """Write one store; read behavior is deliberately unavailable while building."""

    def __init__(
        self,
        path: Path,
        *,
        channels: int,
        dtype: np.dtype | str = "float32",
        source: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise StoreError(f"channels must be a positive integer, got {channels!r}")
        try:
            resolved_dtype = np.dtype(dtype)
            DTypeCode.from_dtype(resolved_dtype)
        except (TypeError, IndexFormatError) as exc:
            raise StoreError(f"dtype {dtype!r} is not supported by the DSio store: {exc}") from exc
        self.path = Path(path)
        self.channels = channels
        self.dtype = resolved_dtype
        self.source = source
        self.attrs = validated_attrs(attrs, "store attrs")
        self._entities: list[Entity] = []
        self._entity_ids: set[str] = set()
        self._offsets: list[int] = [0]
        self._rows = 0
        self.path.mkdir(parents=True, exist_ok=True)
        if any(self.path.iterdir()):
            raise StoreError(f"cannot build store at {self.path}: destination is not empty")
        try:
            self._signal = open(self.path / SIGNAL_FILE, "xb")  # noqa: SIM115
        except (FileExistsError, IsADirectoryError) as exc:
            raise StoreError(
                f"cannot build store at {self.path}: destination is not empty"
            ) from exc
        self._closed = False

    def add(
        self,
        entity_id: str,
        signal: np.ndarray,
        *,
        group: str,
        attrs: dict[str, Any] | None = None,
    ) -> Entity:
        """Append one identified sample; insertion order defines its stable position."""
        if self._closed:
            raise StoreError("builder is closed")
        sample_attrs = validated_attrs(attrs, f"entity {entity_id!r} attrs")
        array = np.ascontiguousarray(signal, dtype=self.dtype)
        if array.ndim != 2 or array.shape[1] != self.channels:
            raise StoreError(
                f"entity {entity_id!r} has shape {array.shape}, expected (n, {self.channels})"
            )
        if array.shape[0] == 0:
            raise StoreError(f"entity {entity_id!r} is empty")
        entity = Entity(
            entity_id=entity_id,
            group=group,
            start_row=self._rows,
            n_rows=int(array.shape[0]),
            data_sha256=sha256_of_bytes(array.tobytes()),
            attrs=sample_attrs,
        )
        if entity.entity_id in self._entity_ids:
            raise StoreError(f"duplicate entity_id {entity.entity_id!r}")

        self._signal.write(array.tobytes())
        self._entities.append(entity)
        self._entity_ids.add(entity.entity_id)
        self._rows += entity.n_rows
        self._offsets.append(self._rows)
        return entity

    def close(self) -> StoreManifest:
        """Flush the payload and publish its index, metadata, and manifest."""
        if self._closed:
            raise StoreError("builder is already closed")
        if not self._entities:
            self._signal.close()
            self._closed = True
            raise StoreError("cannot build a store with no entities")

        try:
            self._signal.flush()
            os.fsync(self._signal.fileno())
        finally:
            self._signal.close()
            self._closed = True

        header = IndexHeader(
            version=FORMAT_VERSION,
            dtype=self.dtype,
            channels=self.channels,
            n_entities=len(self._entities),
            n_rows=self._rows,
        )
        write_index(self.path / INDEX_FILE, header, np.array(self._offsets, dtype=np.int64))

        payload = "\n".join(
            json.dumps(entity.model_dump(mode="json"), sort_keys=True) for entity in self._entities
        )
        atomic_write(self.path / ENTITIES_FILE, (payload + "\n").encode())

        manifest = StoreManifest(
            schema_version=STORE_SCHEMA_VERSION,
            name=self.path.name,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            dtype=str(self.dtype),
            channels=self.channels,
            n_rows=self._rows,
            n_entities=len(self._entities),
            n_groups=len({entity.group for entity in self._entities}),
            signal_sha256=sha256_of_file(str(self.path / SIGNAL_FILE)),
            index_sha256=sha256_of_file(str(self.path / INDEX_FILE)),
            entities_sha256=sha256_of_file(str(self.path / ENTITIES_FILE)),
            signal_bytes=(self.path / SIGNAL_FILE).stat().st_size,
            source=self.source,
            attrs=self.attrs,
        )
        atomic_write(
            self.path / MANIFEST_FILE,
            yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=True).encode(),
        )
        return manifest

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
        elif not self._closed:
            self._signal.close()
            self._closed = True
