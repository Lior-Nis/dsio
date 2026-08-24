"""``Fold`` invariants: the last line of defence for folds that did not come from a
validated split file.

Moved here from ``test_loop.py`` when Task 6 of the fold-as-process plan deleted
``cross_validate`` (and its test file with it). ``Fold`` itself was never in question --
these tests exercise ``Fold.__post_init__`` in ``eval/contract.py``, which is unchanged and
still very much alive.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsio.eval.contract import EvalError, Fold


def test_a_fold_with_overlapping_train_and_test_is_rejected() -> None:
    """The last line of defence for folds that did not come from a validated split file."""
    with pytest.raises(EvalError, match="appear in both"):
        Fold(index=0, train=np.array([0, 1, 2]), test=np.array([2, 3]))


def test_a_fold_that_repeats_a_row_is_rejected() -> None:
    with pytest.raises(EvalError, match="more than once"):
        Fold(index=0, train=np.array([0, 0, 1]), test=np.array([2]))


def test_a_validation_part_must_also_be_disjoint() -> None:
    with pytest.raises(EvalError, match="appear in both"):
        Fold(index=0, train=np.array([0, 1]), test=np.array([2]), val=np.array([1]))


def test_an_empty_test_part_is_rejected() -> None:
    with pytest.raises(EvalError, match="nothing to score"):
        Fold(index=0, train=np.array([0, 1]), test=np.array([], dtype=int))


def test_an_empty_train_part_needs_saying_so_explicitly() -> None:
    """Evaluating something that was not trained here is a real shape — a benchmark pass,
    a shipped model, an agent behind an API — and the alternative is inventing a token
    training set that both lies and violates the disjointness this class enforces."""
    with pytest.raises(EvalError, match="evaluation_only=True"):
        Fold(index=0, train=np.array([], dtype=int), test=np.array([1, 2]))

    fold = Fold(
        index=0, train=np.array([], dtype=int), test=np.array([1, 2]), evaluation_only=True
    )
    assert fold.sizes["train"] == 0


def test_a_fold_names_itself_when_unnamed() -> None:
    assert Fold(index=3, train=np.array([0]), test=np.array([1])).name == "fold3"
