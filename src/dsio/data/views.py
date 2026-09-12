"""Windowed views: an index of offsets, never a copy of the data.

Changing a window length costs seconds and megabytes here, not hours and tens of
gigabytes. Materialising one corpus once per (length, stride, labelling policy) is how a
store reaches hundreds of gigabytes; the same configurations are a few index files over one
copy.

The index is content-addressed by its spec, so a view either already exists for exactly
these parameters or is rebuilt — the same rule a run's ``config_hash`` uses for identity.

**Every window carries its entity and group.** That is not bookkeeping. Overlapping windows
that straddle a split boundary put near-identical rows in train and test simultaneously —
Kapoor & Narayanan's L1.4 and L3.2, and the most common fatal bug in sensor ML. It is
invisible if the index is only offsets, so the index refuses to be only offsets.

**An index also says nothing about what it left behind.** Windowing discards rows by
construction — a recording shorter than one window contributes nothing, and
``drop_last_partial`` truncates every trailing remainder — and an index of offsets has no
place to record either. A store of 8 entities under a 64-row window kept 320 of 634 rows,
three recordings absent entirely, and the examples contract, the store/index binding check,
the dataset and the loader all passed without a word. :func:`window_discards` is where that
arithmetic lives, and :func:`build_index` refuses the unambiguous half of it by default.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from dsio.contracts import DsioModel, sha256_of, short_digest
from dsio.contracts.hashing import sha256_of_bytes
from dsio.data.store import Entity, SignalStore, StoreError

VIEWS_DIRNAME = "views"

LabelPolicy = Literal["any", "majority", "ratio", "none"]
TimeUnit = Literal["row", "epoch_s"]

DroppedEntityPolicy = Literal["raise", "drop"]
"""What :func:`build_index` does about entities too short to hold a single window."""

MAX_NAMED_ENTITIES = 5
"""How many lost entities a refusal lists by name before summarising the rest."""


class ViewError(ValueError):
    """Raised when a view cannot be built as asked, or would quietly not be what it claims.

    A subclass of ``ValueError`` because that is what this module raised before it had a
    type of its own, and :class:`~dsio.data.adapters.SignalExamples` catches ``ValueError``
    to re-raise as ``ExamplesError``. Callers that want to distinguish "this store and this
    spec do not fit each other" from any other bad argument now can.
    """


T_START_ATTR = "t_start"
SAMPLE_RATE_ATTR = "sample_rate"


class WindowSpec(DsioModel):
    """How to slice a store into windows.

    ``dense_stride`` applies a tighter stride only where a mask marks rare positives, so
    class imbalance is addressed at *index* time rather than by duplicating data. It costs
    offsets, not gigabytes.
    """

    length: int = Field(gt=0)
    stride: int = Field(gt=0)
    dense_stride: int | None = Field(default=None, gt=0)
    label_policy: LabelPolicy = "none"
    label_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    drop_last_partial: bool = True
    min_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Per-window metric floors, e.g. {'purity': 0.9}. Part of the digest.",
    )
    max_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Per-window metric ceilings.",
    )

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
    strings. That is not micro-optimisation: at tens of millions of windows, holding an
    entity id and a group string per window costs ~72 bytes each — gigabytes of index, which
    would defeat the entire point of not materialising the windows. Codes plus a table cost
    12 bytes per window.

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
        metrics: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.starts = np.ascontiguousarray(starts, dtype=np.int64)
        self.entity_codes = np.ascontiguousarray(entity_codes, dtype=np.int32)
        self.entity_names = list(entity_names)
        self.entity_groups = list(entity_groups)
        self.spec = spec
        self.store_name = store_name
        self.labels = labels
        self.metrics = dict(metrics or {})
        if self.starts.size != self.entity_codes.size:
            raise ViewError("starts and entity_codes must be the same length")
        if len(self.entity_names) != len(self.entity_groups):
            raise ViewError("entity_names and entity_groups must be the same length")

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
            metrics={name: values[mask] for name, values in self.metrics.items()},
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
        for name, values in self.metrics.items():
            arrays[f"metric__{name}"] = values
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
                metrics={
                    key.removeprefix("metric__"): data[key]
                    for key in data.files
                    if key.startswith("metric__")
                },
                spec=WindowSpec.model_validate(meta["spec"]),
                store_name=meta["store"],
            )

    def __repr__(self) -> str:
        return (
            f"WindowIndex({self.store_name!r}, n={len(self):,}, "
            f"length={self.spec.length}, stride={self.spec.stride}, "
            f"groups={len(set(self.entity_groups))})"
        )


def assert_index_matches_store(store: SignalStore, index: WindowIndex) -> None:
    """Raise if ``index`` was built against a different store than ``store``.

    A window's offset is only meaningful relative to the store it was cut from; an index
    built against one store and read from another would silently return the wrong bytes at
    every position rather than fail. This is a data-layer invariant, checked without torch,
    so anything that turns an index and a store into windows -- torch-facing or not -- can
    call it instead of repeating the comparison.
    """
    if index.store_name != store.path.name:
        raise ViewError(
            f"index was built for store {index.store_name!r}, not {store.path.name!r}"
        )


def window_times(
    store: SignalStore, index: WindowIndex, *, unit: TimeUnit = "row"
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_start, t_end)`` per window in the requested unit.

    ``row`` is the offset *within the window's own entity*, which is the right coordinate
    when each recording is an independent timeline. Using the global row offset instead
    would make entity 2's row 1000 look simultaneous with entity 1's row 1000, and a
    walk-forward cut would slice each recording at a different point in its own history.

    ``epoch_s`` needs ``t_start`` and ``sample_rate`` on each entity, and fails loudly
    without them rather than silently falling back to row order.
    """
    starts = index.starts.astype(np.float64)
    entity_start = np.array(
        [store.entity(name).start_row for name in index.entity_names], dtype=np.float64
    )
    relative = starts - entity_start[index.entity_codes]

    if unit == "row":
        return relative, relative + float(index.spec.length)

    t0 = np.empty(len(index.entity_names), dtype=np.float64)
    rate = np.empty(len(index.entity_names), dtype=np.float64)
    for i, name in enumerate(index.entity_names):
        attrs = store.entity(name).attrs
        if T_START_ATTR not in attrs or SAMPLE_RATE_ATTR not in attrs:
            raise StoreError(
                f"entity {name!r} lacks {T_START_ATTR!r}/{SAMPLE_RATE_ATTR!r}, which "
                f"time_unit='epoch_s' requires; set them at ingest or use time_unit='row'"
            )
        t0[i] = float(attrs[T_START_ATTR])
        rate[i] = float(attrs[SAMPLE_RATE_ATTR])
        if rate[i] <= 0:
            raise StoreError(f"entity {name!r} has non-positive {SAMPLE_RATE_ATTR}")

    seconds = relative / rate[index.entity_codes]
    absolute = t0[index.entity_codes] + seconds
    return absolute, absolute + float(index.spec.length) / rate[index.entity_codes]


@dataclass(frozen=True)
class EntityDiscard:
    """How many of one entity's rows no window of a view covers.

    ``covered_rows`` counts each row once however many windows read it: a stride below the
    window length re-reads rows, and summing window lengths instead would let a view that
    truncates a recording report covering more rows than the recording has.
    """

    entity_id: str
    group: str
    n_rows: int
    covered_rows: int

    @property
    def discarded_rows(self) -> int:
        return self.n_rows - self.covered_rows

    @property
    def is_dropped(self) -> bool:
        """True when the entity contributes no window at all, not merely a short one."""
        return self.covered_rows == 0


@dataclass(frozen=True)
class WindowDiscards:
    """What a ``(store, spec)`` pair leaves out of the index it would build.

    Kept apart from :class:`WindowIndex` rather than attached to it, because an index is
    content-addressed and cached: :func:`load_or_build` returns one that was built now and
    one that was read from disk interchangeably, and a field populated on the first and
    empty on the second would be exactly the sort of quiet inconsistency this accounting
    exists to remove. It is a function of the store and the spec, so any caller can ask for
    it at any time, before or after building.

    The two kinds of loss are not equally serious and are reported separately.
    ``truncated`` is ``drop_last_partial`` doing its job -- a partial trailing window is
    meaningless on a waveform -- and is reported, never refused. ``dropped`` is a whole
    recording missing from the index, which no downstream check can see, and is what
    :func:`build_index` refuses by default.
    """

    store: str
    length: int
    entities: tuple[EntityDiscard, ...]
    n_rows: int
    covered_rows: int

    @property
    def discarded_rows(self) -> int:
        return self.n_rows - self.covered_rows

    @property
    def dropped(self) -> tuple[EntityDiscard, ...]:
        """Entities contributing no window at all, in store order."""
        return tuple(entity for entity in self.entities if entity.is_dropped)

    @property
    def truncated(self) -> tuple[EntityDiscard, ...]:
        """Entities that yield windows but lose a remainder to ``drop_last_partial``."""
        return tuple(entity for entity in self.entities if not entity.is_dropped)

    def __bool__(self) -> bool:
        """Truthy when the view costs rows, so ``if window_discards(...)`` reads plainly."""
        return self.discarded_rows > 0

    def describe(self) -> str:
        """One line: how much of the corpus the view keeps, and what it cost."""
        share = 100.0 * self.covered_rows / self.n_rows if self.n_rows else 100.0
        return (
            f"{self.covered_rows:,}/{self.n_rows:,} rows ({share:.0f}%) covered by windows "
            f"of length {self.length}; {len(self.dropped)} entity(s) dropped entirely, "
            f"{len(self.truncated)} truncated"
        )


def _entity_offsets(
    entity: Entity, spec: WindowSpec, dense_mask: np.ndarray | None
) -> list[int]:
    """Sorted window starts for one entity, or ``[]`` if it cannot hold one window.

    The single enumeration both :func:`build_index` and :func:`window_discards` read, so
    the accounting is of the windows that would actually be built rather than of a
    plausible re-derivation of them -- a dense stride adds windows the plain stride does
    not, and a report computed from the stride alone would overstate the loss.
    """
    span = entity.n_rows - spec.length
    if span < 0:
        return []  # recording shorter than one window
    last = entity.start_row + span

    offsets = set(range(entity.start_row, last + 1, spec.stride))
    if not spec.drop_last_partial and last not in offsets:
        offsets.add(last)

    if dense_mask is not None and spec.dense_stride is not None:
        region = dense_mask[entity.start_row : entity.end_row]
        for pos in range(0, span + 1, spec.dense_stride):
            if region[pos : pos + spec.length].any():
                offsets.add(entity.start_row + pos)

    return sorted(offsets)


def _covered_rows(offsets: Sequence[int], length: int) -> int:
    """Size of the union of ``[offset, offset + length)`` over sorted ``offsets``."""
    covered = 0
    reached = 0
    for offset in offsets:
        end = offset + length
        covered += end - max(offset, reached)
        reached = end
    return covered


def window_discards(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None = None,
    row_metrics: dict[str, np.ndarray] | None = None,
) -> WindowDiscards:
    """Account for every row of ``store`` that ``spec`` would leave out of an index.

    Windowing is lossy by construction and the index cannot say so: it holds offsets, and
    an offset that was never emitted leaves no trace. This is the only place the corpus and
    the view are compared, so it is the only place a caller can learn that a 64-row window
    kept half a store.

    Metric floors (``min_metrics`` / ``max_metrics``) are counted only when ``row_metrics``
    is supplied, and are never a reason to refuse a build. Without them this stays what it
    has always been -- a function of the store and the spec, answerable before anything is
    built or read.

    They are worth counting when they can be. A caller writes
    ``min_metrics={'purity': 0.9}``; they do not write "and remove this subject from the
    study". The filter is deliberate and the spec digest records it; a recording filtered
    away to nothing is neither, and it is the same invisible loss as a recording too short
    to window -- a group missing from every fold while every check passes. It is reported
    rather than refused because filtering a recording to nothing may be exactly what a
    purity floor is for, and refusing would be a guess about how a project filters that
    reporting cannot get wrong.
    """
    entities: list[EntityDiscard] = []
    total_rows = 0
    total_covered = 0
    for entity in store.entities:
        offsets = _entity_offsets(entity, spec, dense_mask)
        if row_metrics is not None and offsets:
            starts = np.asarray(offsets, dtype=np.int64)
            _, keep = _metric_filter(spec, starts, row_metrics)
            offsets = starts[keep].tolist()
        covered = _covered_rows(offsets, spec.length)
        total_rows += entity.n_rows
        total_covered += covered
        if covered < entity.n_rows:
            entities.append(
                EntityDiscard(
                    entity_id=entity.entity_id,
                    group=entity.group,
                    n_rows=entity.n_rows,
                    covered_rows=covered,
                )
            )
    return WindowDiscards(
        store=store.path.name,
        length=spec.length,
        entities=tuple(entities),
        n_rows=total_rows,
        covered_rows=total_covered,
    )


def _refuse_dropped_entities(
    store: SignalStore, spec: WindowSpec, policy: DroppedEntityPolicy
) -> None:
    """Refuse a view that would omit whole recordings, unless the caller asked for that.

    An entity yields no window exactly when it is shorter than one window, so this costs a
    pass over the entity table and never enumerates offsets -- cheap enough to run on
    :func:`load_or_build`'s cache-hit path too, where the loss is otherwise laundered by a
    file an earlier caller left behind.
    """
    if policy == "drop":
        return
    lost = [entity for entity in store.entities if entity.n_rows < spec.length]
    if not lost:
        return

    named = ", ".join(f"{e.entity_id} ({e.n_rows:,} rows)" for e in lost[:MAX_NAMED_ENTITIES])
    if len(lost) > MAX_NAMED_ENTITIES:
        named += f", and {len(lost) - MAX_NAMED_ENTITIES:,} more"
    rows = sum(entity.n_rows for entity in lost)
    longest = max(entity.n_rows for entity in lost)
    raise ViewError(
        f"{len(lost)} of {len(store.entities)} entities in {store.path.name!r} are shorter "
        f"than one {spec.length}-row window and would contribute no windows at all: "
        f"{named}. That is {rows:,} of {store.n_rows:,} rows absent from the index, and an "
        "index of offsets cannot say what it does not contain -- the split, the dataset and "
        "the loader would all pass over a corpus missing these recordings entirely. Lower "
        f"length to {longest:,} or below to keep them, or pass "
        "on_dropped_entities='drop' to state that recordings shorter than a window are "
        "meant to go. Rows lost to drop_last_partial are a separate, smaller matter: "
        "window_discards(store, spec) reports those without refusing them."
    )


def _refuse_multi_window_entities(index: WindowIndex, store: SignalStore) -> None:
    """Refuse an index whose entities are not one-for-one with its windows.

    The hole :func:`_refuse_dropped_entities` leaves open, and the opposite failure. An
    entity *larger* than the window is not lost: it becomes several windows that are not
    items, and nothing downstream can tell. The index is internally consistent, the loader
    yields tensors of the right shape, and every one of them is a crop of an image rather
    than an image -- a corpus of 12x12 pictures read as pairs of 64-row strips, scoring
    perfectly plausibly on something that is not the data.

    Only the caller knows a window was meant to be a whole item, which is why this is a
    claim rather than a default: a waveform corpus wants many windows per recording, and
    that is the normal case.

    Counted from the built index rather than re-derived from the store, so it costs a
    bincount over the windows and works unchanged on :func:`load_or_build`'s cache-hit
    path -- and so it sees the index that will actually be used, including whatever a
    metric floor removed from it.
    """
    counts = np.bincount(index.entity_codes, minlength=len(index.entity_names))
    offenders = [
        (index.entity_names[code], int(n)) for code, n in enumerate(counts) if n != 1
    ]
    if not offenders:
        return

    named = ", ".join(
        f"{name} ({n} windows)" for name, n in offenders[:MAX_NAMED_ENTITIES]
    )
    if len(offenders) > MAX_NAMED_ENTITIES:
        named += f", and {len(offenders) - MAX_NAMED_ENTITIES:,} more"
    sizes = sorted({entity.n_rows for entity in store.entities})
    raise ViewError(
        f"one_window_per_entity was claimed, but {len(offenders)} of "
        f"{len(index.entity_names)} entities in {store.path.name!r} do not yield exactly "
        f"one {index.spec.length}-row window: {named}. An entity larger than the window "
        "becomes several windows that are not items, and an index of offsets cannot say "
        "so -- the loader yields crops of the right shape and nothing downstream can tell "
        f"them from items. The store holds {len(sizes)} distinct entity size(s) "
        f"({', '.join(f'{s:,}' for s in sizes[:5])}"
        f"{', ...' if len(sizes) > 5 else ''}); resize at ingest so every item is the same "
        "length, give each size its own store, or drop the claim and treat these as "
        "windows."
    )


def build_index(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    row_metrics: dict[str, np.ndarray] | None = None,
    on_dropped_entities: DroppedEntityPolicy = "raise",
    one_window_per_entity: bool = False,
) -> WindowIndex:
    """Enumerate window starts for every entity in ``store``.

    Windows never cross an entity boundary: each recording is enumerated independently, so
    a window can never splice the end of one session onto the start of another.

    ``dense_mask`` is a per-row boolean over the whole store marking regions worth sampling
    more tightly (rare positives). ``labels`` is a per-row array used to derive a window
    label under the spec's policy.

    ``row_metrics`` are per-row arrays — annotation purity, sensor validity — averaged over
    each window and then filtered by the spec's ``min_metrics`` / ``max_metrics``. Filtering
    belongs in the spec rather than applied afterwards so it is part of the index digest:
    two indices differing only in a purity floor are different indices, and must not share
    a cache entry, which is what lets a quality floor discard part of a view reproducibly.

    ``on_dropped_entities`` decides what happens to a recording shorter than one window.
    Such a recording cannot be windowed — that much is arithmetic — but it also vanishes
    without trace: the index names only entities it holds windows for, so a store whose
    every short session was silently omitted produces a perfectly consistent index over a
    corpus that is not the one on disk. Refusing by default is the same stance
    :mod:`dsio.eval.pool` takes toward a fold that never ran. ``"drop"`` is the explicit
    opt-out, because deciding that short recordings do not belong in a view is a legitimate
    choice — it just has to be a choice.

    A trailing remainder truncated by ``drop_last_partial`` is *not* refused: a partial
    window is meaningless on a waveform, and discarding one is what the flag is for. It is
    still a cost, and :func:`window_discards` is where a caller reads it — for either kind
    of loss, and without having to opt in to anything.

    ``one_window_per_entity`` claims the corpus is one of items rather than of recordings —
    images, documents, trials — so that every entity must yield exactly one window. It is
    the guarantee whole-item access rests on, and the one thing the checks above cannot
    infer: an entity *larger* than the window is not lost, it becomes several windows that
    are not items, and every downstream check passes on crops. Off by default, because a
    waveform corpus wants many windows per recording. It is a claim about the corpus, not a
    property of the view, so like ``on_dropped_entities`` it stays out of the spec digest.
    """
    _refuse_dropped_entities(store, spec, on_dropped_entities)

    starts: list[int] = []
    codes: list[int] = []
    entity_names = [entity.entity_id for entity in store.entities]
    entity_groups = [entity.group for entity in store.entities]

    for code, entity in enumerate(store.entities):
        for offset in _entity_offsets(entity, spec, dense_mask):
            starts.append(offset)
            codes.append(code)

    start_array = np.array(starts, dtype=np.int64)
    code_array = np.array(codes, dtype=np.int32)

    window_labels = None
    if labels is not None and spec.label_policy != "none":
        window_labels = _derive_labels(np.asarray(labels), start_array, spec)

    window_metrics, keep = _metric_filter(spec, start_array, row_metrics)

    index = WindowIndex(
        starts=start_array[keep],
        entity_codes=code_array[keep],
        entity_names=entity_names,
        entity_groups=entity_groups,
        spec=spec,
        store_name=store.path.name,
        labels=None if window_labels is None else window_labels[keep],
        metrics={name: values[keep] for name, values in window_metrics.items()},
    )
    if one_window_per_entity:
        _refuse_multi_window_entities(index, store)
    return index


def _metric_filter(
    spec: WindowSpec, starts: np.ndarray, row_metrics: dict[str, np.ndarray] | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-window metrics, and the mask the spec's floors and ceilings keep.

    The single filter both :func:`build_index` and :func:`window_discards` read, for the
    same reason :func:`_entity_offsets` is the single enumeration: a report derived from
    its own re-reading of the spec would eventually disagree with the index it claims to
    describe, and the disagreement would be silent.
    """
    window_metrics = {
        name: _window_means(np.asarray(values, dtype=np.float64), starts, spec.length)
        for name, values in (row_metrics or {}).items()
    }
    keep = np.ones(starts.size, dtype=bool)
    for name, floor in spec.min_metrics.items():
        if name not in window_metrics:
            raise ViewError(
                f"spec filters on metric {name!r}, but it was not supplied in row_metrics; "
                f"available: {', '.join(sorted(window_metrics)) or 'none'}"
            )
        keep &= window_metrics[name] >= floor
    for name, ceiling in spec.max_metrics.items():
        if name not in window_metrics:
            raise ViewError(
                f"spec filters on metric {name!r}, but it was not supplied in row_metrics; "
                f"available: {', '.join(sorted(window_metrics)) or 'none'}"
            )
        keep &= window_metrics[name] <= ceiling
    return window_metrics, keep


def _window_means(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Mean of a per-row array over each window, via a cumulative sum.

    O(rows + windows) rather than O(windows x length): at 42M windows the naive loop is
    the difference between seconds and an afternoon.
    """
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    return (cumulative[starts + length] - cumulative[starts]) / length


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


def _normalise_index_inputs(
    spec: WindowSpec,
    dense_mask: np.ndarray | None,
    labels: np.ndarray | None,
    row_metrics: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    """Own one semantic snapshot of every runtime input that changes an index."""
    normalised_mask = (
        None
        if dense_mask is None or spec.dense_stride is None
        else np.array(dense_mask, dtype=bool, order="C", copy=True)
    )
    normalised_labels = (
        None
        if labels is None or spec.label_policy == "none"
        else np.array(labels, dtype=bool, order="C", copy=True)
    )
    normalised_metrics = {
        name: np.array(values, dtype=np.float64, order="C", copy=True)
        for name, values in sorted((row_metrics or {}).items())
    }
    return normalised_mask, normalised_labels, normalised_metrics


def _array_identity(values: np.ndarray | None) -> dict[str, Any] | None:
    if values is None:
        return None
    return {
        "dtype": values.dtype.str,
        "shape": [int(size) for size in values.shape],
        "sha256": sha256_of_bytes(values.tobytes(order="C")),
    }


def _index_cache_identity(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None,
    labels: np.ndarray | None,
    row_metrics: dict[str, np.ndarray],
) -> str:
    """Full content identity for one logically distinct cached index."""
    manifest = store.manifest()
    return sha256_of(
        {
            "algorithm": "window-index:v2",
            "store": {
                "signal": manifest.signal_sha256,
                "index": manifest.index_sha256,
                "entities": manifest.entities_sha256,
            },
            "spec": spec.model_dump(mode="json"),
            "inputs": {
                "dense_mask": _array_identity(dense_mask),
                "labels": _array_identity(labels),
                "row_metrics": {
                    name: _array_identity(values) for name, values in row_metrics.items()
                },
            },
        }
    )


def _index_path_for_inputs(
    store: SignalStore,
    spec: WindowSpec,
    root: Path | None,
    *,
    dense_mask: np.ndarray | None,
    labels: np.ndarray | None,
    row_metrics: dict[str, np.ndarray],
) -> Path:
    base = root or store.path.parent.parent / VIEWS_DIRNAME
    identity = _index_cache_identity(
        store,
        spec,
        dense_mask=dense_mask,
        labels=labels,
        row_metrics=row_metrics,
    )
    return base / store.path.name / f"{identity}.npz"


def index_path(
    store: SignalStore,
    spec: WindowSpec,
    root: Path | None = None,
    *,
    dense_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    row_metrics: dict[str, np.ndarray] | None = None,
) -> Path:
    """Content-addressed location for a built index.

    The identity covers the complete store, the spec, and every runtime input that changes
    the logical index. Policy-only validation flags remain outside it.
    """
    dense_mask, labels, row_metrics = _normalise_index_inputs(
        spec, dense_mask, labels, row_metrics
    )
    return _index_path_for_inputs(
        store,
        spec,
        root,
        dense_mask=dense_mask,
        labels=labels,
        row_metrics=row_metrics,
    )


def load_or_build(
    store: SignalStore,
    spec: WindowSpec,
    *,
    dense_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    row_metrics: dict[str, np.ndarray] | None = None,
    root: Path | None = None,
    on_dropped_entities: DroppedEntityPolicy = "raise",
    one_window_per_entity: bool = False,
) -> WindowIndex:
    """Return the index for ``spec``, building and caching it if absent.

    ``on_dropped_entities`` is checked on the cache-hit path as well as the build path, and
    deliberately does not enter the cache key. The key binds an index to committed splits
    and staged runs, so adding a caller's policy to it would invalidate every one of them;
    but a caller who did not opt in to losing recordings must still be refused when an
    earlier caller who did opt in already left the index on disk, or the cache launders the
    loss. The check is a pass over the entity table, not over the windows.

    ``one_window_per_entity`` is checked on both paths for the same reason and, being
    counted from the index itself, needs no re-derivation to check a cached one.
    """
    dense_mask, labels, row_metrics = _normalise_index_inputs(
        spec, dense_mask, labels, row_metrics
    )
    path = _index_path_for_inputs(
        store,
        spec,
        root,
        dense_mask=dense_mask,
        labels=labels,
        row_metrics=row_metrics,
    )
    if path.is_file():
        _refuse_dropped_entities(store, spec, on_dropped_entities)
        cached = WindowIndex.load(path)
        if one_window_per_entity:
            _refuse_multi_window_entities(cached, store)
        return cached
    index = build_index(
        store,
        spec,
        dense_mask=dense_mask,
        labels=labels,
        row_metrics=row_metrics,
        on_dropped_entities=on_dropped_entities,
        one_window_per_entity=one_window_per_entity,
    )
    index.save(path)
    return index
