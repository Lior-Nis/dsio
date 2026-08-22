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
from dsio.splits.models import SplitError, SplitFile, SplitFold
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
    """Build one :class:`~dsio.eval.contract.Fold` per fold across the given split files.

    One split file now holds a whole family's worth of folds (``SplitFile.folds``), so
    this flattens every fold from every file given, in order. A family committed the old
    way — one file per fold — flattens identically, since each such file just has one
    entry. Fold *index* always comes from the fold's own declared ``index``, never from
    its position in a file's list or in ``splits``: renumbering by position would silently
    scramble the correspondence a run's artifacts and a comparison key off, which matters
    more once running a subset of folds is the normal case, not the exception.
    """
    if not splits:
        raise SplitError("no split files were given; there are no folds to build")

    folds: list[Fold] = []
    for split in splits:
        for split_fold in split.folds:
            masks = resolve_masks(examples, split, split_fold, require_total=require_total)
            for required in (train_part, test_part):
                if required not in masks:
                    raise SplitError(
                        f"split {split.name!r} fold {split_fold.index} has no {required!r} "
                        f"part; it defines {', '.join(sorted(masks)) or 'nothing'}"
                    )
            val = None
            if val_part is not None and val_part in masks:
                val = np.flatnonzero(masks[val_part])
            folds.append(
                Fold(
                    index=split_fold.index,
                    train=np.flatnonzero(masks[train_part]),
                    test=np.flatnonzero(masks[test_part]),
                    val=val,
                    name=f"{split.name}[{split_fold.index}]",
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
            f"no split files under {directory}; commit a split file there first — dsio "
            "reads splits, it does not generate them"
        )

    def ordinal(path: Path) -> int:
        digits = path.stem.removeprefix("fold")
        return int(digits) if digits.isdigit() else -1

    return sorted(found, key=ordinal)


def require_fold(root: Path | str, name: str, index: int) -> SplitFold:
    """Fail before any data loads if ``index`` is not a fold split family ``name`` declares.

    This is the check a task-level ``fold`` field needs at the point the split is
    resolved: purely from the committed YAML, with no store, index or ``Examples`` built
    yet. A family may live in one file holding every fold (the current layout) or the
    still-supported legacy layout of one file per fold, so every file ``fold_paths``
    finds is loaded and its folds pooled before the lookup, rather than checking only the
    first file and missing folds declared in the others.

    The lookup and its message are :meth:`~dsio.splits.models.SplitFile.fold`'s, not a
    second copy of them: ``model_copy`` never re-runs validation, so pooling every file's
    folds onto one file's identity this way is only ever used for this read, not to
    smuggle an unvalidated family past ``SplitFile``'s own checks.
    """
    files = [SplitFile.load(Path(path)) for path in fold_paths(root, name)]
    pooled = files[0].model_copy(update={"folds": [f for file in files for f in file.folds]})
    return pooled.fold(index)


def _assert_test_parts_are_disjoint(folds: Sequence[Fold]) -> None:
    """No window may be tested by two folds.

    Within a fold, disjointness is checked by ``Fold`` itself. Across folds it is a
    different property, and the one that corrupts a pooled out-of-fold metric: an example
    tested twice is counted twice, which quietly reweights the score toward whichever ones
    were duplicated. The loop would catch this too, but catching it here means it fails
    before anything is fitted rather than after the last fold.
    """
    positions = (
        np.concatenate([fold.test for fold in folds]) if folds else np.empty(0, dtype=np.int64)
    )
    values, counts = np.unique(positions, return_counts=True)
    repeated = values[counts > 1]
    if repeated.size:
        owners = {
            int(position): [fold.name for fold in folds if position in set(fold.test.tolist())]
            for position in repeated[:5].tolist()
        }
        detail = "; ".join(f"{pos} in {' and '.join(names)}" for pos, names in owners.items())
        raise SplitError(
            f"{repeated.size} example(s) appear in more than one test part; "
            f"folds must test disjoint examples: {detail}"
        )
