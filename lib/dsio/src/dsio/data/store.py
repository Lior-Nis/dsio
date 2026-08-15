"""The canonical signal store: one immutable copy of a corpus.

A corpus is stored **once**, as continuous signal. Windows are produced by an index of
offsets rather than by copying — see :mod:`dsio.data.views`. FORGE reached 229 GB across 29
Zarr stores because every combination of window length, stride and labelling policy was a
full physical copy; the same corpus here is one store plus a few megabytes of index per
configuration.

Layout::

    <root>/<name>/
        signal.bin       raw C-contiguous [n_rows, channels], the payload
        signal.idx       versioned header + entity row offsets
        entities.jsonl   one row per recording: id, group, offset, length, attrs
        manifest.yaml    content hashes and provenance — committed to git

The store is **immutable once built**. That is what lets readers run lock-free across
worker processes, and it makes a content hash meaningful: a store either matches its
manifest or it is corrupt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import yaml
from pydantic import Field

from dsio.contracts import DsioModel, atomic_write, sha256_of_file
from dsio.data.format import IndexHeader, read_index, write_index
from dsio.data.readers import SignalReader, open_reader

DATA_ROOT_ENV = "DSIO_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("stores")

SIGNAL_FILE = "signal.bin"
INDEX_FILE = "signal.idx"
ENTITIES_FILE = "entities.jsonl"
MANIFEST_FILE = "manifest.yaml"


class StoreError(RuntimeError):
    """Raised when a store is malformed, incomplete, or contradicts its manifest."""


def data_root() -> Path:
    """Where stores live. Bulk data does not belong in the repository."""
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


class Entity(DsioModel):
    """One contiguous recording inside the payload.

    ``group`` is the leakage boundary, and it is required rather than optional. Splitting
    on entities that share a subject, machine or symbol is how near-identical rows end up
    on both sides of a split; making the caller name the group forces the question to be
    answered at ingest, when the answer is known.
    """

    entity_id: str
    group: str
    start_row: int
    n_rows: int
    attrs: dict[str, Any] = Field(default_factory=dict)

    @property
    def end_row(self) -> int:
        return self.start_row + self.n_rows


class StoreManifest(DsioModel):
    """Committed description of a store. The payload itself is gitignored."""

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


@dataclass(frozen=True, slots=True)
class Window:
    """A single window: the signal, and enough provenance to place it."""

    data: np.ndarray
    entity_id: str
    group: str
    start_row: int


class SignalStoreBuilder:
    """Writes a store. Deliberately separate from the read path.

    Megatron keeps its builder out of the read path for the same reason: a reader that can
    also write is a reader that can corrupt a store someone else is mapping.
    """

    def __init__(
        self,
        path: Path,
        *,
        channels: int,
        dtype: np.dtype | str = "float32",
        source: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.channels = channels
        self.dtype = np.dtype(dtype)
        self.source = source
        self.attrs = attrs or {}
        self._entities: list[Entity] = []
        self._offsets: list[int] = [0]
        self._rows = 0
        self.path.mkdir(parents=True, exist_ok=True)
        # Held open across many add() calls, so the builder itself is the context
        # manager rather than this handle. Closed in close() and in __exit__.
        self._signal = open(self.path / SIGNAL_FILE, "wb")  # noqa: SIM115
        self._closed = False

    def add(
        self,
        entity_id: str,
        signal: np.ndarray,
        *,
        group: str,
        attrs: dict[str, Any] | None = None,
    ) -> Entity:
        """Append one recording. Order is preserved and defines the row offsets."""
        if self._closed:
            raise StoreError("builder is closed")
        array = np.ascontiguousarray(signal, dtype=self.dtype)
        if array.ndim != 2 or array.shape[1] != self.channels:
            raise StoreError(
                f"entity {entity_id!r} has shape {array.shape}, expected (n, {self.channels})"
            )
        if array.shape[0] == 0:
            raise StoreError(f"entity {entity_id!r} is empty")
        if any(existing.entity_id == entity_id for existing in self._entities):
            raise StoreError(f"duplicate entity_id {entity_id!r}")

        entity = Entity(
            entity_id=entity_id,
            group=group,
            start_row=self._rows,
            n_rows=int(array.shape[0]),
            attrs=attrs or {},
        )
        self._signal.write(array.tobytes())
        self._entities.append(entity)
        self._rows += entity.n_rows
        self._offsets.append(self._rows)
        return entity

    def close(self) -> StoreManifest:
        """Finalise: flush the payload, write the index, entities and manifest."""
        if self._closed:
            raise StoreError("builder is already closed")
        if not self._entities:
            raise StoreError("cannot build a store with no entities")

        self._signal.flush()
        os.fsync(self._signal.fileno())
        self._signal.close()
        self._closed = True

        header = IndexHeader(
            version=1,
            dtype=self.dtype,
            channels=self.channels,
            n_entities=len(self._entities),
            n_rows=self._rows,
        )
        write_index(self.path / INDEX_FILE, header, np.array(self._offsets, dtype=np.int64))

        payload = "\n".join(
            json.dumps(entity.model_dump(mode="json"), sort_keys=True)
            for entity in self._entities
        )
        atomic_write(self.path / ENTITIES_FILE, (payload + "\n").encode("utf-8"))

        manifest = StoreManifest(
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


class SignalStore:
    """Read access to a canonical store.

    Readers are opened lazily and cached per process, so an instance can be constructed in
    a parent and used inside DataLoader workers under either fork or spawn.
    """

    def __init__(self, path: Path | str, *, backend: str = "mmap") -> None:
        self.path = Path(path)
        self.backend = backend
        index_path = self.path / INDEX_FILE
        if not index_path.is_file():
            raise StoreError(f"no store at {self.path}: {INDEX_FILE} is missing")
        self.header, self.offsets = read_index(index_path)
        self.entities: list[Entity] = [
            Entity.model_validate(json.loads(line))
            for line in (self.path / ENTITIES_FILE).read_text().splitlines()
            if line.strip()
        ]
        self._by_id = {entity.entity_id: entity for entity in self.entities}
        self._readers: dict[int, SignalReader] = {}

    @classmethod
    def builder(cls, path: Path | str, **kwargs: Any) -> SignalStoreBuilder:
        return SignalStoreBuilder(Path(path), **kwargs)

    @classmethod
    def open(cls, name: str, *, root: Path | None = None, backend: str = "mmap") -> SignalStore:
        """Open a store by name under the data root."""
        return cls((root or data_root()) / name, backend=backend)

    @property
    def _reader(self) -> SignalReader:
        """The reader for this process, opened on first use.

        Keyed by pid so a forked worker never shares the parent's mapping, and a spawned
        one never tries to unpickle it.
        """
        pid = os.getpid()
        reader = self._readers.get(pid)
        if reader is None:
            reader = open_reader(
                self.backend,
                self.path / SIGNAL_FILE,
                self.header.dtype,
                self.header.channels,
                self.header.n_rows,
            )
            self._readers[pid] = reader
        return reader

    def __getstate__(self) -> dict[str, Any]:
        """Drop live readers when pickled; the child reopens its own."""
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

    def entity(self, entity_id: str) -> Entity:
        try:
            return self._by_id[entity_id]
        except KeyError:
            raise StoreError(f"no entity {entity_id!r} in store {self.path.name}") from None

    def read(self, start: int, n_rows: int) -> np.ndarray:
        """Read ``n_rows`` from global row ``start``, refusing to cross an entity boundary.

        A window spanning two recordings is silently wrong — it splices the end of one
        session onto the start of another and trains on a discontinuity that never
        happened. Checking here costs one binary search.
        """
        if start < 0 or n_rows <= 0:
            raise StoreError(f"invalid read: start={start}, n_rows={n_rows}")
        if start + n_rows > self.header.n_rows:
            raise StoreError(
                f"read [{start}, {start + n_rows}) exceeds store length {self.header.n_rows}"
            )
        idx = int(np.searchsorted(self.offsets, start, side="right") - 1)
        if start + n_rows > int(self.offsets[idx + 1]):
            entity = self.entities[idx]
            raise StoreError(
                f"read [{start}, {start + n_rows}) crosses the end of entity "
                f"{entity.entity_id!r} at row {entity.end_row}; windows must stay within "
                "one recording"
            )
        return self._reader.read_rows(start, n_rows)

    def read_entity(self, entity_id: str) -> np.ndarray:
        entity = self.entity(entity_id)
        return self.read(entity.start_row, entity.n_rows)

    def entity_at(self, row: int) -> Entity:
        """Which recording a global row belongs to."""
        if row < 0 or row >= self.header.n_rows:
            raise StoreError(f"row {row} is outside the store")
        return self.entities[int(np.searchsorted(self.offsets, row, side="right") - 1)]

    def manifest(self) -> StoreManifest:
        path = self.path / MANIFEST_FILE
        if not path.is_file():
            raise StoreError(f"store {self.path.name} has no {MANIFEST_FILE}")
        return StoreManifest.model_validate(yaml.safe_load(path.read_text()))

    def verify(self) -> None:
        """Re-hash every file and compare against the manifest, failing closed.

        Existence is not integrity. Kedro's ``--only-missing-outputs`` treats a truncated
        file as valid forever; a store that has silently lost bytes is worse than one that
        is obviously absent.
        """
        manifest = self.manifest()
        for filename, expected in (
            (SIGNAL_FILE, manifest.signal_sha256),
            (INDEX_FILE, manifest.index_sha256),
            (ENTITIES_FILE, manifest.entities_sha256),
        ):
            actual = sha256_of_file(str(self.path / filename))
            if actual != expected:
                raise StoreError(
                    f"{self.path.name}/{filename} has digest {actual[:12]}, manifest says "
                    f"{expected[:12]}; the store has been modified or corrupted"
                )

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


def list_stores(root: Path | None = None) -> Sequence[str]:
    base = root or data_root()
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob(f"*/{MANIFEST_FILE}"))
