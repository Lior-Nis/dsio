"""Validated, identity-preserving reads from the canonical store."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from dsio.contracts import sha256_of_bytes, sha256_of_file
from dsio.data.format import IndexFormatError, IndexHeader, read_index
from dsio.data.readers import SignalReader, open_reader
from dsio.data.store.builder import SignalStoreBuilder
from dsio.data.store.layout import (
    DATA_ROOT_ENV,
    DEFAULT_DATA_ROOT,
    ENTITIES_FILE,
    INDEX_FILE,
    MANIFEST_FILE,
    SIGNAL_FILE,
    STORE_SCHEMA_VERSION,
    Entity,
    StoredSample,
    StoreError,
    StoreManifest,
)


def data_root() -> Path:
    """Return the configured store root without creating it."""
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


class SignalStore:
    """The single public read interface for immutable numeric samples and row windows."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.header, self.offsets = self._read_index()
        self.entities = self._read_entities()
        self._by_id = {entity.entity_id: entity for entity in self.entities}
        self._readers: dict[int, SignalReader] = {}
        self._validate_layout(self._read_manifest())

    @classmethod
    def builder(cls, path: Path | str, **kwargs: Any) -> SignalStoreBuilder:
        return SignalStoreBuilder(Path(path), **kwargs)

    @classmethod
    def open(cls, name: str, *, root: Path | None = None) -> SignalStore:
        """Open a named store under the configured or explicit root."""
        return cls((root or data_root()) / name)

    @property
    def _reader(self) -> SignalReader:
        pid = os.getpid()
        reader = self._readers.get(pid)
        if reader is None:
            reader = open_reader(
                self.path / SIGNAL_FILE,
                self.header.dtype,
                self.header.channels,
                self.header.n_rows,
            )
            self._readers[pid] = reader
        return reader

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    @property
    def n_rows(self) -> int:
        return self.header.n_rows

    @property
    def channels(self) -> int:
        return self.header.channels

    @property
    def groups(self) -> list[str]:
        return sorted({entity.group for entity in self.entities})

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(entity.entity_id for entity in self.entities)

    def entity(self, entity_id: str) -> Entity:
        try:
            return self._by_id[entity_id]
        except KeyError:
            raise StoreError(f"no entity {entity_id!r} in store {self.path.name}") from None

    def read(self, start: int, n_rows: int) -> np.ndarray:
        """Read rows without crossing a persisted sample boundary."""
        if start < 0 or n_rows <= 0:
            raise StoreError(f"invalid read: start={start}, n_rows={n_rows}")
        if start + n_rows > self.header.n_rows:
            raise StoreError(
                f"read [{start}, {start + n_rows}) exceeds store length {self.header.n_rows}"
            )
        index = int(np.searchsorted(self.offsets, start, side="right") - 1)
        if start + n_rows > int(self.offsets[index + 1]):
            entity = self.entities[index]
            raise StoreError(
                f"read [{start}, {start + n_rows}) crosses the end of entity "
                f"{entity.entity_id!r} at row {entity.end_row}; windows must stay within "
                "one recording"
            )
        return self._reader.read_rows(start, n_rows)

    def read_sample(self, key: str | int) -> StoredSample:
        """Read one whole persisted sample by stable identity or zero-based position."""
        entity = self._sample_entity(key)
        data = self.read(entity.start_row, entity.n_rows)
        actual = sha256_of_bytes(np.ascontiguousarray(data).tobytes())
        if actual != entity.data_sha256:
            raise StoreError(
                f"sample {entity.entity_id!r} in store {self.path.name!r} has data digest "
                f"{actual[:12]}, expected {entity.data_sha256[:12]}; signal.bin is corrupt"
            )
        return {
            "sample_id": entity.entity_id,
            "data": data,
            "group": entity.group,
            "attrs": dict(entity.attrs),
        }

    def read_entity(self, entity_id: str) -> np.ndarray:
        """Compatibility name for the content-checked identity read."""
        return self.read_sample(entity_id)["data"]

    def entity_at(self, row: int) -> Entity:
        if row < 0 or row >= self.header.n_rows:
            raise StoreError(f"row {row} is outside the store")
        return self.entities[int(np.searchsorted(self.offsets, row, side="right") - 1)]

    def manifest(self) -> StoreManifest:
        manifest = self._read_manifest()
        self._validate_layout(manifest)
        return manifest

    def verify(self) -> None:
        """Hash all committed files and fail if any content changed."""
        manifest = self.manifest()
        for filename, expected in (
            (SIGNAL_FILE, manifest.signal_sha256),
            (INDEX_FILE, manifest.index_sha256),
            (ENTITIES_FILE, manifest.entities_sha256),
        ):
            try:
                actual = sha256_of_file(str(self.path / filename))
            except OSError as exc:
                raise StoreError(f"cannot verify {self.path / filename}: {exc}") from exc
            if actual != expected:
                raise StoreError(
                    f"{self.path.name}/{filename} has digest {actual[:12]}, manifest says "
                    f"{expected[:12]}; the store has been modified or corrupted"
                )

    def _sample_entity(self, key: str | int) -> Entity:
        if isinstance(key, bool):
            raise StoreError("sample key must be a string identity or non-negative position")
        if isinstance(key, str):
            try:
                return self._by_id[key]
            except KeyError:
                raise StoreError(
                    f"no sample identity {key!r} in store {self.path.name!r}"
                ) from None
        if isinstance(key, int):
            if key < 0 or key >= len(self.entities):
                raise StoreError(
                    f"sample position {key} is outside store {self.path.name!r} with "
                    f"{len(self.entities)} samples"
                )
            return self.entities[key]
        raise StoreError(
            f"sample key must be a string identity or non-negative position, got "
            f"{type(key).__name__}"
        )

    def _read_index(self) -> tuple[IndexHeader, np.ndarray]:
        path = self.path / INDEX_FILE
        if not path.is_file():
            raise StoreError(f"no store at {self.path}: {INDEX_FILE} is missing")
        try:
            return read_index(path)
        except (OSError, IndexFormatError) as exc:
            raise StoreError(f"cannot read index {path}: {exc}") from exc

    def _read_entities(self) -> list[Entity]:
        path = self.path / ENTITIES_FILE
        if not path.is_file():
            raise StoreError(f"store {self.path.name!r} has no {ENTITIES_FILE}")
        try:
            return [
                Entity.model_validate(json.loads(line))
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise StoreError(f"cannot read entity metadata {path}: {exc}") from exc

    def _read_manifest(self) -> StoreManifest:
        path = self.path / MANIFEST_FILE
        if not path.is_file():
            raise StoreError(f"store {self.path.name!r} has no {MANIFEST_FILE}")
        try:
            return StoreManifest.model_validate(yaml.safe_load(path.read_text()))
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise StoreError(f"cannot read manifest {path}: {exc}") from exc

    def _validate_layout(self, manifest: StoreManifest) -> None:
        if manifest.schema_version != STORE_SCHEMA_VERSION:
            raise StoreError(
                f"store {self.path.name!r} declares schema version "
                f"{manifest.schema_version}, expected {STORE_SCHEMA_VERSION}"
            )
        payload = self.path / SIGNAL_FILE
        if not payload.is_file():
            raise StoreError(f"store {self.path.name!r} has no {SIGNAL_FILE}")
        try:
            actual_bytes = payload.stat().st_size
        except OSError as exc:
            raise StoreError(f"cannot inspect {payload}: {exc}") from exc
        if actual_bytes != self.header.payload_bytes:
            raise StoreError(
                f"{self.path.name}/{SIGNAL_FILE} has {actual_bytes} bytes, expected "
                f"{self.header.payload_bytes} from {INDEX_FILE}"
            )
        expected_header = {
            "dtype": str(self.header.dtype),
            "channels": self.header.channels,
            "n_rows": self.header.n_rows,
            "n_entities": self.header.n_entities,
        }
        for field, expected in expected_header.items():
            actual = getattr(manifest, field)
            if actual != expected:
                raise StoreError(
                    f"store {self.path.name!r} manifest {field}={actual!r}, but "
                    f"{INDEX_FILE} requires {expected!r}"
                )
        if manifest.signal_bytes != actual_bytes:
            raise StoreError(
                f"store {self.path.name!r} manifest signal_bytes={manifest.signal_bytes}, "
                f"but {SIGNAL_FILE} has {actual_bytes} bytes"
            )
        if len(self.entities) != self.header.n_entities:
            raise StoreError(
                f"store {self.path.name!r} has {len(self.entities)} entity records, "
                f"expected {self.header.n_entities} from {INDEX_FILE}"
            )
        seen: set[str] = set()
        for position, entity in enumerate(self.entities):
            start = int(self.offsets[position])
            rows = int(self.offsets[position + 1] - start)
            if entity.start_row != start:
                raise StoreError(
                    f"entity {entity.entity_id!r} start_row={entity.start_row}, expected "
                    f"{start} from {INDEX_FILE}"
                )
            if entity.n_rows != rows:
                raise StoreError(
                    f"entity {entity.entity_id!r} n_rows={entity.n_rows}, expected "
                    f"{rows} from {INDEX_FILE}"
                )
            if entity.entity_id in seen:
                raise StoreError(f"duplicate sample identity {entity.entity_id!r}")
            seen.add(entity.entity_id)
            _require_sha256(entity.data_sha256, f"entity {entity.entity_id!r} data_sha256")
        groups = len({entity.group for entity in self.entities})
        if manifest.n_groups != groups:
            raise StoreError(
                f"store {self.path.name!r} manifest n_groups={manifest.n_groups}, "
                f"but entity metadata has {groups}"
            )
        for filename, expected in (
            (INDEX_FILE, manifest.index_sha256),
            (ENTITIES_FILE, manifest.entities_sha256),
        ):
            _require_sha256(expected, f"manifest {filename} digest")
            path = self.path / filename
            try:
                actual = sha256_of_file(str(path))
            except OSError as exc:
                raise StoreError(f"cannot validate {path}: {exc}") from exc
            if actual != expected:
                raise StoreError(
                    f"{self.path.name}/{filename} has digest {actual[:12]}, manifest says "
                    f"{expected[:12]}; metadata is corrupt"
                )
        _require_sha256(manifest.signal_sha256, "manifest signal digest")

    def __iter__(self) -> Iterator[Entity]:
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def __repr__(self) -> str:
        return (
            f"SignalStore({self.path.name!r}, rows={self.header.n_rows:,}, "
            f"channels={self.header.channels}, entities={len(self.entities)}, "
            f"groups={len(self.groups)})"
        )


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64:
        raise StoreError(f"{field} is not a 64-character SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError:
        raise StoreError(f"{field} is not hexadecimal") from None


def list_stores(root: Path | None = None) -> Sequence[str]:
    base = root or data_root()
    if not base.is_dir():
        return []
    return sorted(path.parent.name for path in base.glob(f"*/{MANIFEST_FILE}"))
