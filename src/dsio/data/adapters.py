"""Concrete :class:`~dsio.data.examples.Examples` implementations.

Three, spanning the shapes dsio is for: a table of independent-ish rows, a windowed signal
corpus, and a set of records identified by key. None of them is privileged — the split
layer cannot tell them apart, which is the point.

A project adds its own by satisfying the protocol; nothing here needs to know about it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dsio.contracts import sha256_of_bytes
from dsio.data.examples import ExamplesError
from dsio.data.views import window_times


class TableExamples:
    """Rows of a table, with a grouping column and any number of attributes.

    The adapter for tabular work, and the one to copy when writing a new one. It holds no
    features at all: the runner already has the dataframe, and dsio only needs to know how
    the rows may be divided.
    """

    def __init__(
        self,
        *,
        name: str,
        groups: Sequence[Any] | np.ndarray,
        attributes: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
        times: tuple[np.ndarray, np.ndarray] | None = None,
        digest: str | None = None,
    ) -> None:
        self._name = name
        self._groups = np.asarray(groups)
        self._attributes = {
            key: np.asarray(value) for key, value in (attributes or {}).items()
        }
        self._times = times
        self._digest = digest or self._derive_digest()

    def _derive_digest(self) -> str:
        """Identity from the grouping and attributes, not from the features.

        Two tables with identical structure and different feature values are the same
        *division* of the data, so a split computed against one applies to the other. That
        is the right granularity here: a split file binds to how the rows are grouped, and
        rebuilding it because an unrelated column changed would be noise.
        """
        payload = self._groups.tobytes() + b"".join(
            key.encode() + np.asarray(value).tobytes()
            for key, value in sorted(self._attributes.items())
        )
        return sha256_of_bytes(payload)[:16]

    @property
    def name(self) -> str:
        return self._name

    @property
    def digest(self) -> str:
        return self._digest

    def __len__(self) -> int:
        return int(self._groups.size)

    @property
    def groups(self) -> np.ndarray:
        return self._groups

    def attribute_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._attributes))

    def attribute(self, name: str) -> np.ndarray:
        try:
            return self._attributes[name]
        except KeyError:
            raise ExamplesError(
                f"{self._name} has no attribute {name!r}; it has "
                f"{', '.join(self.attribute_names()) or 'none'}"
            ) from None

    def times(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._times

    def subset(self, mask: np.ndarray) -> TableExamples:
        mask = np.asarray(mask, dtype=bool)
        return TableExamples(
            name=self._name,
            groups=self._groups[mask],
            attributes={key: value[mask] for key, value in self._attributes.items()},
            times=None if self._times is None else (self._times[0][mask], self._times[1][mask]),
            digest=self._digest,
        )

    def __repr__(self) -> str:
        groups = len(set(self._groups.tolist()))
        return f"TableExamples({self._name!r}, n={len(self):,}, groups={groups})"


class SignalExamples:
    """Windows over a memory-mapped signal corpus — one adapter, not the abstraction.

    Everything time-series-specific lives on this side of the protocol: the window index,
    the entity boundaries, the sample rate. The split layer sees the same seven members it
    sees for a table, so nothing above knows this modality exists.
    """

    def __init__(self, store: Any, index: Any, *, time_unit: str = "row") -> None:
        if index.store_name != store.path.name:
            raise ExamplesError(
                f"index was built for store {index.store_name!r}, not {store.path.name!r}"
            )
        self.store = store
        self.index = index
        self.time_unit = time_unit

    @property
    def name(self) -> str:
        return str(self.store.path.name)

    @property
    def digest(self) -> str:
        """The corpus, not the view of it.

        A split names groups, and which groups exist is a property of the corpus rather than
        of any window length taken over it. So a split generated before a view existed
        resolves against every view of the same data, and re-indexing at a different window
        length does not invalidate a committed split.

        That is deliberate and it is the useful direction: splits outlive views. Binding to
        the view instead would force a window length to be chosen before the split, which is
        backwards — and would silently invalidate a published split whenever anyone tried a
        different stride.
        """
        try:
            return str(self.store.manifest().signal_sha256[:16])
        except Exception:  # noqa: BLE001 - a store without a manifest is still splittable
            return "nomanifest"

    def __len__(self) -> int:
        return len(self.index)

    @property
    def groups(self) -> np.ndarray:
        return self.index.groups

    def attribute_names(self) -> tuple[str, ...]:
        names = set(self.index.metrics)
        if self.index.labels is not None:
            names.add("label")
        for entity in self.store.entities:
            names.update(entity.attrs)
        return tuple(sorted(names))

    def attribute(self, name: str) -> np.ndarray:
        """Window-level metrics first, then the label, then entity attributes broadcast.

        Entity attributes are broadcast to their windows rather than the reverse, because a
        subject-level fact — age, site, protocol — is true of every window that subject
        produced, and stratifying on it is the common case.
        """
        if name in self.index.metrics:
            return np.asarray(self.index.metrics[name])
        if name == "label" and self.index.labels is not None:
            return np.asarray(self.index.labels)

        per_entity = np.array(
            [entity.attrs.get(name, np.nan) for entity in self.store.entities], dtype=object
        )
        if all(value is np.nan or value != value for value in per_entity):
            raise ExamplesError(
                f"{self.name} has no attribute {name!r}; it has "
                f"{', '.join(self.attribute_names()) or 'none'}"
            )
        return per_entity[self.index.entity_codes]

    def times(self) -> tuple[np.ndarray, np.ndarray] | None:
        return window_times(self.store, self.index, unit=self.time_unit)  # type: ignore[arg-type]

    def subset(self, mask: np.ndarray) -> SignalExamples:
        return SignalExamples(self.store, self.index.subset(mask), time_unit=self.time_unit)

    def covered_rows(self) -> np.ndarray:
        """Every raw row these windows touch — the signal-specific overlap proof.

        Not part of the protocol. Overlapping windows are a hazard of this modality
        specifically: two windows that straddle a split boundary put near-identical rows on
        both sides, which a group check alone cannot see. A table has no equivalent, so
        this lives here rather than being a member every adapter has to implement.
        """
        return self.index.covered_rows()

    def __repr__(self) -> str:
        return f"SignalExamples({self.name!r}, n={len(self):,})"


def entity_examples(store: Any) -> TableExamples:
    """One example per recording in a store, for splitting before any view exists.

    Generating a group split does not need windows: the assignment is over groups, and a
    recording already knows which group it belongs to. Building the index first would force
    a window length to be chosen before the split, which is backwards — the split outlives
    every view taken of it.
    """
    entities = list(store.entities)
    names = sorted({key for entity in entities for key in entity.attrs})
    try:
        digest = store.manifest().signal_sha256[:16]
    except Exception:  # noqa: BLE001 - a store without a manifest is still splittable
        digest = None
    return TableExamples(
        name=str(store.path.name),
        groups=[entity.group for entity in entities],
        attributes={
            key: [entity.attrs.get(key, np.nan) for entity in entities] for key in names
        },
        digest=digest,
    )
