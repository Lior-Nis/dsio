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

    One committed split file holds a whole family's worth of folds (``SplitFile.folds``);
    this flattens every fold from every ``SplitFile`` given, in order — the callers below
    always give exactly one, loaded from the family's single committed file, but this stays
    general over ``SplitFile`` objects a caller built directly (tests do, to isolate a
    subset of folds). Fold *index* always comes from the fold's own declared ``index``,
    never from its position in a file's list or in ``splits``: renumbering by position
    would silently scramble the correspondence a run's artifacts and a comparison key off,
    which matters more once running a subset of folds is the normal case, not the
    exception.
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
    path: Path | str,
    **kwargs: object,
) -> list[Fold]:
    """Load the split file at ``path`` and build folds from it."""
    files = [SplitFile.load(Path(path))]
    return folds_from_splits(examples, files, **kwargs)  # type: ignore[arg-type]


def split_path(root: Path | str, name: str) -> Path:
    """The one committed split file for a split family.

    One ``SplitFile`` holds every fold in a family (``SplitFile.folds``, an ordered
    list) — that is the only layout dsio reads. There used to be a second, glob-based
    layout here, one file per fold sorted by the fold number parsed from its filename;
    it is gone rather than kept alongside this one, because lexical sorting of those
    filenames gave ten folds back as ``0, 1, 10, 2`` — an ordering bug that produced a
    perfectly plausible result with the folds silently permuted, undetectable downstream
    because every individual fold was still valid. A single committed file removes the
    ordering question rather than continuing to guard against it.
    """
    candidate = Path(root) / name / "split.yaml"
    if not candidate.is_file():
        raise SplitError(
            f"no split file at {candidate}; commit a split file there first — dsio "
            "reads splits, it does not generate them"
        )
    return candidate


def require_fold(root: Path | str, name: str, index: int) -> SplitFold:
    """Fail before any data loads if ``index`` is not a fold split family ``name`` declares.

    This is the check a task-level ``fold`` field needs at the point the split is
    resolved: purely from the committed YAML, with no store, index or ``Examples`` built
    yet. The lookup and its message are :meth:`~dsio.splits.models.SplitFile.fold`'s, not
    a second copy of them.
    """
    return SplitFile.load(split_path(root, name)).fold(index)


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
