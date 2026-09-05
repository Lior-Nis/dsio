"""Apply a split file to a dataset.

Splits are resolved **on the fly**: one dataset, one small YAML of group ids, and a boolean
mask per part. Nothing is copied, so a fold costs a mask rather than a dataset.

Disjointness is structural rather than incidental, which is the whole reason the design is
safe. Every example belongs to exactly one group, and split parts are mutually disjoint over
groups, so no example can appear in two parts.

Some modalities carry a second hazard the group check cannot see: examples that overlap in
an underlying buffer, such as sliding windows over a signal. Those datasets expose
``covered_rows()`` and :func:`assert_no_row_overlap` proves the stronger property directly.
It is deliberately not part of the protocol, because for a table or a set of documents there
is nothing underneath to overlap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dsio.data.examples import Examples
from dsio.splits.models import SplitError, SplitFile, SplitFold
from dsio.splits.temporal import apply as apply_temporal


def resolve(
    examples: Examples,
    split: SplitFile,
    fold: SplitFold,
    *,
    require_total: bool = True,
) -> dict[str, Examples]:
    """Divide a dataset into one subset per part, for one fold of ``split``.

    ``require_total`` rejects a split that does not account for every group present.
    Silently dropping examples is how a fold quietly trains on less data than its name
    claims — but it applies to the *group* partition only. A temporal split deliberately
    discards the purged and embargoed band, and that is the point of it.
    """
    masks = resolve_masks(examples, split, fold, require_total=require_total)
    return {part: examples.subset(mask) for part, mask in masks.items()}


def resolve_masks(
    examples: Examples,
    split: SplitFile,
    fold: SplitFold,
    *,
    require_total: bool = True,
) -> dict[str, np.ndarray]:
    """Boolean mask per part, over the dataset's examples, for one fold of ``split``.

    The mask form is what :func:`dsio.splits.folds.folds_from_splits` turns into a
    :class:`~dsio.eval.contract.Fold`'s integer positions; a subset has forgotten where
    its examples came from, so the mask is kept as the intermediate form.
    :func:`resolve` is this plus one ``subset`` call, so both views apply exactly the same
    validation and there is no second code path to keep in step.

    The store binding (``store`` / ``store_manifest_sha256``) lives on ``split``, not on
    ``fold`` — one family binds to one corpus, so that check runs once per call rather than
    being duplicated onto every fold.
    """
    if split.store != examples.name:
        raise SplitError(
            f"split {split.name!r} was built for {split.store!r}, not {examples.name!r}"
        )
    if split.store_manifest_sha256 is not None:
        actual = examples.digest
        if actual != split.store_manifest_sha256:
            raise SplitError(
                f"split {split.name!r} was computed against digest "
                f"{split.store_manifest_sha256[:12]}, but {examples.name!r} is now "
                f"{actual[:12]}; regenerate the split or restore the data"
            )
    # The digest check above cannot catch this one. A subset carries its parent's digest --
    # correctly, since it is the same corpus -- so a split computed on a filtered
    # subpopulation passes every check above and then scores a population it was never
    # computed for. `derivation` is the field that distinguishes them.
    if split.examples_derivation is not None:
        actual_derivation = examples.derivation
        if actual_derivation != split.examples_derivation:
            raise SplitError(
                f"split {split.name!r} was computed against derivation "
                f"{split.examples_derivation}, but {examples.name!r} is "
                f"{actual_derivation}; one of them is a subset of the other, so the "
                "groups this split assigns were chosen over a different population"
            )

    present = {str(g) for g in examples.groups}
    named = fold.all_groups

    unknown = named - present
    if unknown:
        raise SplitError(
            f"split {split.name!r} fold {fold.index} names {len(unknown)} group(s) absent "
            f"from the index: {', '.join(sorted(unknown)[:5])}"
        )
    if require_total and fold.parts:
        unassigned = present - named
        if unassigned:
            raise SplitError(
                f"split {split.name!r} fold {fold.index} does not assign {len(unassigned)} "
                f"group(s) present in the index: {', '.join(sorted(unassigned)[:5])}"
            )

    groups = np.asarray([str(g) for g in examples.groups])
    parts = set(fold.parts) | set(fold.temporal.spans if fold.temporal else ())

    times: tuple[np.ndarray, np.ndarray] | None = None
    if fold.temporal is not None:
        times = examples.times()
        if times is None:
            raise SplitError(
                f"split {split.name!r} fold {fold.index} has temporal bounds, but "
                f"{examples.name!r} has no time coordinates to apply them to"
            )

    out: dict[str, np.ndarray] = {}
    for part in sorted(parts):
        mask = np.ones(len(examples), dtype=bool)
        # A part named only in `temporal` spans every group; the time bounds alone
        # decide it. That is what makes a purely temporal split expressible.
        if fold.parts and part in fold.parts:
            mask &= np.isin(groups, list(fold.parts[part]))
        if fold.temporal is not None and times is not None and part in fold.temporal.spans:
            mask &= apply_temporal(fold.temporal, *times, part=part)
        out[part] = mask
    return out


def assert_no_row_overlap(parts: dict[str, Any]) -> None:
    """Prove no underlying row appears in two parts.

    For modalities whose examples overlap in a shared buffer. Expensive by construction — it
    materialises every covered row — so it belongs in tests and in an explicit check
    command, not in the training path.
    """
    for part, subset in parts.items():
        if not hasattr(subset, "covered_rows"):
            raise SplitError(
                f"part {part!r} cannot prove row-level disjointness: its dataset has no "
                "covered_rows(). This check is specific to modalities whose examples "
                "overlap in an underlying buffer, such as windowed signal."
            )
    covered: dict[str, np.ndarray] = {
        part: subset.covered_rows() for part, subset in parts.items()
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


def summarise(parts: dict[str, Examples]) -> dict[str, dict[str, int]]:
    """Example and group counts per part, for logging into the run record."""
    return {
        part: {
            "examples": len(subset),
            "groups": len({str(g) for g in subset.groups}),
        }
        for part, subset in parts.items()
    }
