"""Turn committed split files into folds the loop can run.

This is the seam that joins the two halves of dsio: a dataset and its committed split
files on one side, the fold loop and the artifact contract on the other. Neither side knows
what modality it is handling.

A fold is integer positions into an :class:`~dsio.data.examples.Examples`, so nothing is
copied: cross-validating a large corpus costs one position array per part, not one dataset
per fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dsio.data.examples import Examples
from dsio.eval.contract import Fold
from dsio.splits.models import SplitError, SplitFile
from dsio.splits.resolve import resolve_masks

TRAIN_PART = "train"
TEST_PART = "test"
VAL_PART = "val"


def folds_from_splits(
    examples: Examples,
    splits: Sequence[SplitFile],
    *,
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
        masks = resolve_masks(examples, split, require_total=require_total)
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
    examples: Examples,
    paths: Sequence[Path | str],
    **kwargs: object,
) -> list[Fold]:
    """Load split files from disk in the order given and build folds from them."""
    files = [SplitFile.load(Path(path)) for path in paths]
    return folds_from_splits(examples, files, **kwargs)  # type: ignore[arg-type]


def fold_paths(root: Path | str, name: str) -> list[Path]:
    """Every fold file committed under a split name, in fold order.

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
    different property, and the one that corrupts a pooled out-of-fold metric: an example
    tested twice is counted twice, which quietly reweights the score toward whichever ones
    were duplicated. The loop would catch this too, but catching it here means it fails
    before anything is fitted rather than after the last fold.
    """
    seen: dict[int, str] = {}
    for fold in folds:
        for position in fold.test.tolist():
            if position in seen:
                raise SplitError(
                    f"example {position} is in the test part of both {seen[position]!r} "
                    f"and {fold.name!r}; folds must test disjoint examples"
                )
            seen[position] = fold.name
