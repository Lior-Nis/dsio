"""Turn committed split files into folds the loop can run.

This is the seam that joins the two halves of dsio: the store, the window index and the
split YAMLs on one side, the fold loop and the artifact contract on the other. Everything
either side of it was independently tested before this module existed; this is where the
design is actually load-bearing.

A fold is integer positions into a :class:`~dsio.data.views.WindowIndex`, so nothing is
copied — a five-fold cross-validation over 42M windows costs five pairs of index arrays,
not five datasets.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.eval.contract import Fold
from dsio.splits.models import SplitError, SplitFile
from dsio.splits.resolve import resolve_masks

TRAIN_PART = "train"
TEST_PART = "test"
VAL_PART = "val"


def folds_from_splits(
    index: WindowIndex,
    splits: Sequence[SplitFile],
    *,
    store: SignalStore | None = None,
    train_part: str = TRAIN_PART,
    test_part: str = TEST_PART,
    val_part: str | None = VAL_PART,
    require_total: bool = True,
) -> list[Fold]:
    """Build one :class:`~dsio.eval.contract.Fold` per split file.

    Order is the order given. Fold *index* comes from each file's own ``fold`` field where
    it has one, so a fold number in an artifact always means the same thing as the fold
    number in the committed YAML — renumbering them by list position would silently
    scramble that correspondence when folds are run as a subset.
    """
    if not splits:
        raise SplitError("no split files were given; there are no folds to build")

    folds: list[Fold] = []
    for position, split in enumerate(splits):
        masks = resolve_masks(index, split, store=store, require_total=require_total)
        for required in (train_part, test_part):
            if required not in masks:
                raise SplitError(
                    f"split {split.name!r} has no {required!r} part; it defines "
                    f"{', '.join(sorted(masks)) or 'nothing'}"
                )
        val = None
        if val_part is not None and val_part in masks:
            val = np.flatnonzero(masks[val_part])
        folds.append(
            Fold(
                index=split.fold if split.fold is not None else position,
                train=np.flatnonzero(masks[train_part]),
                test=np.flatnonzero(masks[test_part]),
                val=val,
                name=split.name if split.fold is None else f"{split.name}[{split.fold}]",
            )
        )
    _assert_test_parts_are_disjoint(folds)
    return folds


def load_folds(
    index: WindowIndex,
    paths: Sequence[Path | str],
    *,
    store: SignalStore | None = None,
    **kwargs: object,
) -> list[Fold]:
    """Load split files from disk in the order given and build folds from them."""
    files = [SplitFile.load(Path(path)) for path in paths]
    return folds_from_splits(index, files, store=store, **kwargs)  # type: ignore[arg-type]


def fold_paths(root: Path | str, name: str) -> list[Path]:
    """Every fold file written by ``write_splits`` for a split name, in fold order.

    Sorted by the fold number parsed from the filename rather than lexically, so ten folds
    do not come back as 0, 1, 10, 2 — an ordering bug that produces a perfectly plausible
    result with the folds silently permuted, and which no assertion downstream can detect
    because every fold is individually valid.
    """
    directory = Path(root) / name
    found = list(directory.glob("fold*.yaml"))
    if not found:
        single = directory / "split.yaml"
        if single.is_file():
            return [single]
        raise SplitError(
            f"no split files under {directory}; generate them with `dsio splits make`"
        )

    def ordinal(path: Path) -> int:
        digits = path.stem.removeprefix("fold")
        return int(digits) if digits.isdigit() else -1

    return sorted(found, key=ordinal)


def _assert_test_parts_are_disjoint(folds: Sequence[Fold]) -> None:
    """No window may be tested by two folds.

    Within a fold, disjointness is checked by ``Fold`` itself. Across folds it is a
    different property, and the one that corrupts a pooled out-of-fold metric: a window
    tested twice is counted twice, which quietly reweights the score toward whichever rows
    were duplicated. The loop would catch this too, but catching it here means it fails
    before anything is fitted rather than after the last fold.
    """
    seen: dict[int, str] = {}
    for fold in folds:
        for position in fold.test.tolist():
            if position in seen:
                raise SplitError(
                    f"window {position} is in the test part of both {seen[position]!r} "
                    f"and {fold.name!r}; folds must test disjoint windows"
                )
            seen[position] = fold.name
