"""Windowed views: an index of offsets, never a copy of the data.

Changing a window length costs seconds and megabytes here, not hours and tens of
gigabytes. FORGE's 229 GB across 29 Zarr stores was one corpus materialised once per
(length, stride, labelling policy); the same configurations are a few index files over one
store.

The index is content-addressed by its spec, so a view either already exists for exactly
these parameters or is rebuilt — the same rule the run ledger uses for identity.

**Every window carries its entity and group.** That is not bookkeeping. Overlapping windows
that straddle a split boundary put near-identical rows in train and test simultaneously —
Kapoor & Narayanan's L1.4 and L3.2, and the most common fatal bug in sensor ML. It is
invisible if the index is only offsets, so the index refuses to be only offsets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.contracts import DsioModel, short_digest
from dsio.data.store import SignalStore

VIEWS_DIRNAME = "views"

LabelPolicy = Literal["any", "majority", "ratio", "none"]


class WindowSpec(DsioModel):
    """How to slice a store into windows.

    ``dense_stride`` reproduces FORGE's ``fog_stride_len``: a tighter stride applied only
    where a mask marks rare positives, so class imbalance is addressed at *index* time
    rather than by duplicating data. It costs offsets, not gigabytes.
    """

    length: int = Field(gt=0)
    stride: int = Field(gt=0)
    dense_stride: int | None = Field(default=None, gt=0)
    label_policy: LabelPolicy = "none"
    label_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    drop_last_partial: bool = True

    @model_validator(mode="after")
    def _check(self) -> WindowSpec:
        if self.dense_stride is not None and self.dense_stride > self.stride:
            raise ValueError(
                f"dense_stride {self.dense_stride} must be <= stride {self.stride}; "
                "a denser stride is the point"
            )
        return self

    @property
    def digest(self) -> str:
        return short_digest(self.model_dump(mode="json"))


class WindowIndex:
    """Window start offsets plus the provenance needed to split them safely.

    Entities are stored as integer codes into a name table rather than as repeated
    strings. That is not micro-optimisation: FORGE's unlabelled corpus is 42.2M windows,
    and holding an entity id and a group string per window costs ~72 bytes each — about
    3 GB of index, which would defeat the entire point of not materialising the windows.
    Codes plus a table cost 12 bytes per window.

    Group is derived from entity rather than stored, because an entity belongs to exactly
    one group by construction. Storing both would let them disagree.
    """

    def __init__(
        self,
        *,
        starts: np.ndarray,
        entity_codes: np.ndarray,
        entity_names: Sequence[str],
        entity_groups: Sequence[str],
        spec: WindowSpec,
        store_name: str,
        labels: np.ndarray | None = None,
    ) -> None:
        self.starts = np.ascontiguousarray(starts, dtype=np.int64)
        self.entity_codes = np.ascontiguousarray(entity_codes, dtype=np.int32)
        self.entity_names = list(entity_names)
        self.entity_groups = list(entity_groups)
        self.spec = spec
        self.store_name = store_name
        self.labels = labels
        if self.starts.size != self.entity_codes.size:
            raise ValueError("starts and entity_codes must be the same length")
        if len(self.entity_names) != len(self.entity_groups):
            raise ValueError("entity_names and entity_groups must be the same length")

    def __len__(self) -> int:
        return int(self.starts.size)

    @property
    def digest(self) -> str:
        return self.spec.digest

    @property
    def entity_ids(self) -> np.ndarray:
        """Per-window entity id, decoded on demand."""
        return np.asarray(self.entity_names, dtype=object)[self.entity_codes]

    @property
    def groups(self) -> np.ndarray:
        """Per-window group. Derived from the entity, never stored alongside it."""
        return np.asarray(self.entity_groups, dtype=object)[self.entity_codes]

    @property
    def group_codes(self) -> np.ndarray:
        """Integer group per window, for splitters that only need identity."""
        table = {name: i for i, name in enumerate(sorted(set(self.entity_groups)))}
        per_entity = np.array([table[g] for g in self.entity_groups], dtype=np.int32)
        return per_entity[self.entity_codes]

    def subset(self, mask: np.ndarray) -> WindowIndex:
        """Restrict to a boolean mask, keeping every parallel array aligned."""
        mask = np.asarray(mask, dtype=bool)
        return WindowIndex(
            starts=self.starts[mask],
            entity_codes=self.entity_codes[mask],
            entity_names=self.entity_names,
            entity_groups=self.entity_groups,
            spec=self.spec,
            store_name=self.store_name,
            labels=None if self.labels is None else self.labels[mask],
        )

    def covered_rows(self) -> np.ndarray:
        """Every raw row this index touches, as a sorted unique array.

        Used to prove two splits do not overlap. Only tractable for modest indices, so it
        is a check you run in a test rather than on every epoch.
        """
        if self.starts.size == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.arange(self.spec.length, dtype=np.int64)
        return np.unique((self.starts[:, None] + offsets[None, :]).ravel())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"starts": self.starts, "entity_codes": self.entity_codes}
        if self.labels is not None:
            arrays["labels"] = self.labels
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "store": self.store_name,
                    "spec": self.spec.model_dump(mode="json"),
                    "entity_names": self.entity_names,
                    "entity_groups": self.entity_groups,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, path: Path) -> WindowIndex:
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as data:
            return cls(
                starts=data["starts"],
                entity_codes=data["entity_codes"],
                entity_names=meta["entity_names"],
                entity_groups=meta["entity_groups"],
                labels=data.get("labels"),
                spec=WindowSpec.model_validate(meta["spec"]),
                store_name=meta["store"],
            )

    def __repr__(self) -> str:
        return (
            f"WindowIndex({self.store_name!r}, n={len(self):,}, "
            f"length={self.spec.length}, stride={self.spec.stride}, "
            f"groups={len(set(self.entity_groups))})"
        )


def build_index(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> WindowIndex:
    """Enumerate window starts for every entity in ``store``.

    Windows never cross an entity boundary: each recording is enumerated independently, so
    a window can never splice the end of one session onto the start of another.

    ``dense_mask`` is a per-row boolean over the whole store marking regions worth sampling
    more tightly (rare positives). ``labels`` is a per-row array used to derive a window
    label under the spec's policy.
    """
    starts: list[int] = []
    codes: list[int] = []
    entity_names = [entity.entity_id for entity in store.entities]
    entity_groups = [entity.group for entity in store.entities]

    for code, entity in enumerate(store.entities):
        span = entity.n_rows - spec.length
        if span < 0:
            continue  # recording shorter than one window
        last = entity.start_row + span

        offsets = set(range(entity.start_row, last + 1, spec.stride))
        if not spec.drop_last_partial and last not in offsets:
            offsets.add(last)

        if dense_mask is not None and spec.dense_stride is not None:
            region = dense_mask[entity.start_row : entity.end_row]
            for pos in range(0, span + 1, spec.dense_stride):
                if region[pos : pos + spec.length].any():
                    offsets.add(entity.start_row + pos)

        for offset in sorted(offsets):
            starts.append(offset)
            codes.append(code)

    window_labels = None
    if labels is not None and spec.label_policy != "none":
        window_labels = _derive_labels(np.asarray(labels), np.array(starts, np.int64), spec)

    return WindowIndex(
        starts=np.array(starts, dtype=np.int64),
        entity_codes=np.array(codes, dtype=np.int32),
        entity_names=entity_names,
        entity_groups=entity_groups,
        spec=spec,
        store_name=store.path.name,
        labels=window_labels,
    )


def _derive_labels(labels: np.ndarray, starts: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Reduce per-row labels to one value per window under the spec's policy."""
    out = np.empty(starts.size, dtype=np.float32)
    for i, start in enumerate(starts):
        window = labels[start : start + spec.length]
        ratio = float(np.count_nonzero(window)) / spec.length
        if spec.label_policy == "any":
            out[i] = float(ratio > 0.0)
        elif spec.label_policy == "majority":
            out[i] = float(ratio >= spec.label_threshold)
        else:
            out[i] = ratio
    return out


class WindowView:
    """Reads windows on demand. The thing a DataLoader wraps."""

    def __init__(self, store: SignalStore, index: WindowIndex) -> None:
        if index.store_name != store.path.name:
            raise ValueError(
                f"index was built for store {index.store_name!r}, "
                f"not {store.path.name!r}"
            )
        self.store = store
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.store.read(int(self.index.starts[i]), self.index.spec.length)

    def label(self, i: int) -> float | None:
        return None if self.index.labels is None else float(self.index.labels[i])

    def group(self, i: int) -> str:
        return self.index.entity_groups[int(self.index.entity_codes[i])]

    def __repr__(self) -> str:
        return f"WindowView({self.store.path.name!r}, n={len(self):,})"


def index_path(store: SignalStore, spec: WindowSpec, root: Path | None = None) -> Path:
    """Content-addressed location for a built index."""
    base = root or store.path.parent.parent / VIEWS_DIRNAME
    return base / store.path.name / f"{spec.digest}.npz"


def load_or_build(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    root: Path | None = None,
) -> WindowIndex:
    """Return the index for ``spec``, building and caching it if absent."""
    path = index_path(store, spec, root)
    if path.is_file():
        return WindowIndex.load(path)
    index = build_index(store, spec, dense_mask=dense_mask, labels=labels)
    index.save(path)
    return index
