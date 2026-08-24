"""Pooling N single-fold runs back into one out-of-fold score.

Decision 6 moved cross-validation out of the process, so the checks the in-process fold
loop made while accumulating now have to be made when the files are read back. These tests
exist for those checks, not for the arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.eval.contract import PREDICTIONS_FILE, EvalError
from dsio.eval.pool import pool_folds

DIGEST = "a" * 64


def write_fold(
    directory: Path,
    fold: int,
    rows: list[int],
    *,
    y_score: list[float] | None = None,
    split: str = "fam",
    digest: str = DIGEST,
) -> Path:
    """One run's `predictions.npz`, shaped exactly as `_write_predictions` writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "row_id": np.asarray(rows, dtype=np.int64),
        "fold": np.asarray(fold, dtype=np.int64),
        "y_true": np.asarray([r % 2 for r in rows], dtype=np.int64),
        "y_pred": np.asarray([r % 2 for r in rows], dtype=np.int64),
        "split": np.asarray(split),
        "split_digest": np.asarray(digest),
    }
    if y_score is not None:
        arrays["y_score"] = np.asarray(y_score, dtype=np.float64)
    np.savez_compressed(directory / PREDICTIONS_FILE, **arrays)
    return directory


def test_pooling_two_folds_scores_every_row_once(tmp_path: Path) -> None:
    a = write_fold(tmp_path / "a", 0, [0, 1, 2])
    b = write_fold(tmp_path / "b", 1, [3, 4, 5])
    pooled = pool_folds([a, b], metrics=("accuracy",))
    assert pooled.row_id.tolist() == [0, 1, 2, 3, 4, 5]
    assert pooled.fold.tolist() == [0, 0, 0, 1, 1, 1]
    assert pooled.metrics["accuracy"] == 1.0


def test_folds_pool_in_index_order_not_argument_order(tmp_path: Path) -> None:
    """Ten folds must not pool as 0, 1, 10, 2.

    The caller globs these files, so argument order is whatever the filesystem returned.
    Sorting by the recorded fold index rather than by position makes the pooled arrays
    reproducible; a permuted pooling produces a perfectly plausible number that no
    downstream assertion can detect, because every fold is individually valid.
    """
    late = write_fold(tmp_path / "late", 10, [10])
    early = write_fold(tmp_path / "early", 2, [2])
    pooled = pool_folds([late, early], metrics=("accuracy",))
    assert pooled.folds == (2, 10)
    assert pooled.row_id.tolist() == [2, 10]


def test_a_row_predicted_by_two_folds_is_rejected(tmp_path: Path) -> None:
    """Guard 2. The split already proves each group is tested once; this catches the other
    case -- a correct split whose row positions a runner reported wrongly."""
    a = write_fold(tmp_path / "a", 0, [0, 1, 2])
    b = write_fold(tmp_path / "b", 1, [2, 3])
    with pytest.raises(EvalError, match="predicted by fold 0 and again by fold 1"):
        pool_folds([a, b], metrics=("accuracy",))


def test_scores_from_only_some_folds_are_rejected(tmp_path: Path) -> None:
    """Guard 3, the case that must fail."""
    a = write_fold(tmp_path / "a", 0, [0, 1], y_score=[0.9, 0.1])
    b = write_fold(tmp_path / "b", 1, [2, 3])
    with pytest.raises(EvalError, match="mixture of"):
        pool_folds([a, b], metrics=("accuracy",))


def test_scores_that_are_all_zero_still_pool(tmp_path: Path) -> None:
    """Guard 3, the case that must NOT fail -- and the reason the writer omits the key.

    A fold that genuinely scored every row 0.0 has recorded scores. Had the writer stored
    zeros for a fold with no scores, this case and the one above would be indistinguishable
    and the guard could not be both correct and useful.
    """
    a = write_fold(tmp_path / "a", 0, [0, 1], y_score=[0.0, 0.0])
    b = write_fold(tmp_path / "b", 1, [2, 3], y_score=[0.0, 0.0])
    pooled = pool_folds([a, b], metrics=("accuracy",))
    assert pooled.y_score is not None
    assert pooled.y_score.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_folds_from_different_split_families_are_refused(tmp_path: Path) -> None:
    a = write_fold(tmp_path / "a", 0, [0, 1], split="fam")
    b = write_fold(tmp_path / "b", 1, [2, 3], split="other")
    with pytest.raises(EvalError, match="refusing to pool"):
        pool_folds([a, b], metrics=("accuracy",))


def test_folds_from_different_store_snapshots_are_refused(tmp_path: Path) -> None:
    """The same family recomputed against a re-staged store is not the same experiment."""
    a = write_fold(tmp_path / "a", 0, [0, 1], digest="a" * 64)
    b = write_fold(tmp_path / "b", 1, [2, 3], digest="b" * 64)
    with pytest.raises(EvalError, match="refusing to pool"):
        pool_folds([a, b], metrics=("accuracy",))


def test_a_fold_that_never_ran_is_named(tmp_path: Path) -> None:
    a = write_fold(tmp_path / "a", 0, [0, 1])
    with pytest.raises(EvalError, match="does not exist"):
        pool_folds([a, tmp_path / "missing"], metrics=("accuracy",))


def test_pooling_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="at least one fold"):
        pool_folds([], metrics=("accuracy",))


def test_scores_survive_pooling_exactly_not_approximately(tmp_path: Path) -> None:
    """A threshold sweep on rounded scores finds the wrong one.

    Moved here from the deleted `test_loop.py`, which proved the same property of
    `OutOfFold.save`/`.load` directly. Under fold-as-process the npz round trip happens in
    two places instead of one -- the writer in `torch_task._write_predictions` and the
    reader here in `pool_folds` -- so this is where it is re-proven now that both exist.
    """
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    a = write_fold(tmp_path / "a", 0, list(range(50)), y_score=scores.tolist())
    pooled = pool_folds([a], metrics=("accuracy",))
    assert pooled.y_score is not None
    assert np.array_equal(pooled.y_score, scores)
