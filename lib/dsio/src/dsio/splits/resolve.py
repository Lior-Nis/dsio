"""Apply a split file to a window index.

Splits are resolved **on the fly** against the memory-mapped store: one corpus, one small
YAML of group IDs, and a boolean mask per part. Nothing is copied, so a fold costs a mask
rather than a dataset.

Why row-level overlap does not need checking at resolve time — the property is structural
rather than incidental, and it is worth stating because it is the whole reason this design
is safe:

1. A window never crosses an entity boundary (:meth:`SignalStore.read` refuses).
2. Every entity belongs to exactly one group (the store's ``Entity.group``).
3. Split parts are mutually disjoint over groups (:class:`SplitFile` validates it).

Therefore no raw row can appear in two parts. :func:`assert_no_row_overlap` verifies the
conclusion directly for tests and for `dsio splits check`, but the guarantee comes from the
three invariants, not from a runtime scan — which would be O(windows x length) and
unaffordable on 42M windows.
"""

from __future__ import annotations

import numpy as np

from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.splits.models import SplitError, SplitFile


def resolve(
    index: WindowIndex,
    split: SplitFile,
    *,
    store: SignalStore | None = None,
    require_total: bool = True,
) -> dict[str, WindowIndex]:
    """Split a window index into one sub-index per part.

    ``require_total`` rejects a split that does not account for every group present in the
    index. Silently dropping windows is how a fold quietly trains on less data than its
    name claims.
    """
    if store is not None and split.store != store.path.name:
        raise SplitError(
            f"split {split.name!r} was built for store {split.store!r}, "
            f"not {store.path.name!r}"
        )
    if store is not None and split.store_manifest_sha256 is not None:
        actual = store.manifest().signal_sha256
        if actual != split.store_manifest_sha256:
            raise SplitError(
                f"split {split.name!r} was computed against store digest "
                f"{split.store_manifest_sha256[:12]}, but {store.path.name!r} is now "
                f"{actual[:12]}; regenerate the split or restore the store"
            )

    present = set(index.entity_groups)
    named = split.all_groups

    unknown = named - present
    if unknown:
        raise SplitError(
            f"split {split.name!r} names {len(unknown)} group(s) absent from the index: "
            f"{', '.join(sorted(unknown)[:5])}"
        )
    if require_total:
        unassigned = present - named
        if unassigned:
            raise SplitError(
                f"split {split.name!r} does not assign {len(unassigned)} group(s) present "
                f"in the index: {', '.join(sorted(unassigned)[:5])}"
            )

    groups = index.groups
    out: dict[str, WindowIndex] = {}
    for part, members in split.parts.items():
        mask = np.isin(groups, list(members))
        out[part] = index.subset(mask)
    return out


def assert_no_row_overlap(parts: dict[str, WindowIndex]) -> None:
    """Prove no raw row appears in two parts.

    Expensive by construction — it materialises every covered row — so this belongs in
    tests and in an explicit check command, not in the training path. The invariants in
    this module's docstring are what make it redundant at runtime; this function is how we
    keep verifying that they actually hold.
    """
    covered: dict[str, np.ndarray] = {
        part: index.covered_rows() for part, index in parts.items()
    }
    names = sorted(covered)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = np.intersect1d(covered[left], covered[right], assume_unique=True)
            if shared.size:
                raise SplitError(
                    f"parts {left!r} and {right!r} share {shared.size} raw row(s), "
                    f"first at {int(shared[0])}; windows are leaking across the split"
                )


def summarise(parts: dict[str, WindowIndex]) -> dict[str, dict[str, int]]:
    """Window and group counts per part, for logging into the run record."""
    return {
        part: {
            "windows": len(index),
            "groups": len(set(index.groups.tolist())),
            "entities": len(set(index.entity_codes.tolist())),
        }
        for part, index in parts.items()
    }
