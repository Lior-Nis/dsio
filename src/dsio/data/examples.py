"""What dsio needs from a dataset in order to split, fold and evaluate it.

This is the whole contract. Everything above it — splits, the fold loop, the verdict
machinery, the matrix — works against these members and nothing else, so a corpus of
sensor windows, a table of rows, a set of documents and a batch of agent episodes are all
first-class without any of them appearing in dsio's own types.

The protocol is deliberately about **identity and grouping, not content**. dsio never needs
to see an example's features: it needs to know how many there are, which ones may not be
separated, what can be balanced across folds, and where they sit in time. Reading the actual
data is the runner's business, and keeping it out of here is what stops one modality's shape
from becoming the abstraction.

Structural rather than inherited, matching :class:`~dsio.data.readers.SignalReader`:
a project can satisfy it without importing anything from dsio, which means a dataset
type dsio has never heard of needs no adapter registered anywhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class ExamplesError(ValueError):
    """Raised when a dataset cannot answer something the split layer needs."""


@runtime_checkable
class Examples(Protocol):
    """A set of examples dsio can divide without knowing what they are."""

    @property
    def name(self) -> str:
        """Stable identifier, used to bind a split file to the data it was computed for."""
        ...

    @property
    def digest(self) -> str:
        """Content identity. A split computed against a different digest is refused."""
        ...

    def __len__(self) -> int: ...

    @property
    def groups(self) -> np.ndarray:
        """Per-example group: the leakage boundary.

        The coarsest key whose examples may be near-duplicates of each other — subject,
        machine, well, symbol, document, conversation. It is required rather than optional
        because a dataset with no grouping is a claim that every example is independent,
        and that claim should be made explicitly rather than by omission.
        """
        ...

    def attribute_names(self) -> tuple[str, ...]:
        """Which per-example attributes exist, for stratification and reporting."""
        ...

    def attribute(self, name: str) -> np.ndarray:
        """Per-example values for ``name``, one row per example."""
        ...

    def times(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-example ``(start, end)`` coordinates, or ``None`` if the data is not temporal.

        Returning ``None`` is what makes purged/embargoed splitting unavailable rather than
        silently wrong. A dataset that has an order but no meaningful clock should say so
        by returning ``None``, not by inventing row indices as timestamps.
        """
        ...

    def subset(self, mask: np.ndarray) -> Examples:
        """Restrict to a boolean mask, preserving identity and every parallel array."""
        ...


def group_attribute(examples: Examples, name: str) -> dict[str, float]:
    """Reduce a per-example attribute to one value per group.

    Groups are what get assigned to folds, so stratification needs group-level values. The
    reduction is a mean for numeric attributes and the single distinct value for constant
    ones — and an attribute that varies *within* a group is rejected rather than averaged,
    because averaging a categorical site code across a subject who moved between sites
    produces a number that means nothing and balances nothing.
    """
    values = np.asarray(examples.attribute(name))
    groups = np.asarray(examples.groups)
    out: dict[str, float] = {}
    for group in np.unique(groups):
        selected = values[groups == group]
        if selected.dtype.kind in "fiu":
            out[str(group)] = float(np.mean(selected))
            continue
        distinct = np.unique(selected)
        if distinct.size > 1:
            raise ExamplesError(
                f"attribute {name!r} takes {distinct.size} values within group "
                f"{group!r}; a non-numeric attribute must be constant per group to be "
                "stratified on, or it balances nothing"
            )
        out[str(group)] = float(_as_number(distinct[0]))
    return out


def _as_number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # Categorical levels are hashed to a stable number only for reporting; the
        # stratifier keys on the raw value, so collisions here cannot affect a split.
        return float(abs(hash(value)) % (2**31))


def check(examples: object) -> Examples:
    """Verify an object satisfies the protocol, with a message that says what is missing.

    ``runtime_checkable`` only checks that the members exist, which is enough to catch the
    common mistake — passing a raw array or a torch Dataset — while staying cheap. It
    cannot check the shapes, so :func:`assert_consistent` does that separately.
    """
    missing = [
        member
        for member in (
            "name",
            "digest",
            "groups",
            "attribute_names",
            "attribute",
            "times",
            "subset",
        )
        if not hasattr(examples, member)
    ]
    if missing or not hasattr(examples, "__len__"):
        raise ExamplesError(
            f"{type(examples).__name__} is not an Examples: missing "
            f"{', '.join(missing) or '__len__'}. dsio needs identity and grouping, not "
            "features — see dsio.data.examples.Examples."
        )
    return examples  # type: ignore[return-value]


def assert_consistent(examples: Examples) -> None:
    """Check the parallel arrays actually line up.

    Cheap, and worth doing at the boundary: a groups array one element short of the dataset
    silently shifts every group assignment by one from that point on, which produces a
    perfectly plausible split whose leakage boundary is wrong.
    """
    size = len(examples)
    groups = np.asarray(examples.groups)
    if groups.size != size:
        raise ExamplesError(
            f"{examples.name}: {groups.size} group labels for {size} examples"
        )
    for name in examples.attribute_names():
        values = np.asarray(examples.attribute(name))
        if values.shape[0] != size:
            raise ExamplesError(
                f"{examples.name}: attribute {name!r} has {values.shape[0]} rows for "
                f"{size} examples"
            )
    times = examples.times()
    if times is not None:
        start, end = times
        if start.size != size or end.size != size:
            raise ExamplesError(
                f"{examples.name}: time coordinates have {start.size}/{end.size} entries "
                f"for {size} examples"
            )
        if np.any(end < start):
            raise ExamplesError(f"{examples.name}: some examples end before they start")
