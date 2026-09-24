"""Row-position fold invariants belong to the split boundary."""

from __future__ import annotations

import numpy as np
import pytest

from dsio.data.splits.folds import Fold
from dsio.data.splits.models import SplitError


def test_overlapping_parts_are_rejected() -> None:
    with pytest.raises(SplitError, match="appear in both"):
        Fold(index=0, train=np.array([0, 1, 2]), test=np.array([2, 3]))

    with pytest.raises(SplitError, match="appear in both"):
        Fold(index=0, train=np.array([0, 1]), test=np.array([2]), val=np.array([1]))


def test_repeated_position_is_rejected() -> None:
    with pytest.raises(SplitError, match="more than once"):
        Fold(index=0, train=np.array([0, 0, 1]), test=np.array([2]))


def test_empty_test_is_rejected() -> None:
    with pytest.raises(SplitError, match="nothing to score"):
        Fold(index=0, train=np.array([0, 1]), test=np.array([], dtype=int))


def test_empty_train_requires_an_evaluation_only_fold() -> None:
    with pytest.raises(SplitError, match="evaluation_only=True"):
        Fold(index=0, train=np.array([], dtype=int), test=np.array([1, 2]))

    fold = Fold(
        index=0,
        train=np.array([], dtype=int),
        test=np.array([1, 2]),
        evaluation_only=True,
    )
    assert fold.train.size == 0


def test_unnamed_fold_gets_a_stable_name() -> None:
    assert Fold(index=3, train=np.array([0]), test=np.array([1])).name == "fold3"
