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
WINDOW_DIGEST = "w" * 12
CONFIG_IDENTITY = "c" * 64


def write_fold(
    directory: Path,
    fold: int,
    rows: list[int],
    *,
    y_score: list[float] | None = None,
    split: str = "fam",
    digest: str = DIGEST,
    window_digest: str = WINDOW_DIGEST,
    config_identity: str = CONFIG_IDENTITY,
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
        "window_digest": np.asarray(window_digest),
        "config_identity": np.asarray(config_identity),
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


def test_folds_with_different_window_digests_are_refused(tmp_path: Path) -> None:
    """Critical 1, axis 1. Three folds built from three different `WindowSpec`s -- the
    reviewer's real-run example was `stride=256` for fold 0 and `stride=128` for folds 1-2
    -- give `row_id` a different meaning in each file, so pooling them is not merely
    imprecise, it is meaningless: row 5 in one file and row 5 in another do not name the
    same window."""
    a = write_fold(tmp_path / "a", 0, [0, 1], window_digest="w" * 12)
    b = write_fold(tmp_path / "b", 1, [2, 3], window_digest="x" * 12)
    with pytest.raises(EvalError, match="window"):
        pool_folds([a, b], metrics=("accuracy",))


def test_folds_with_different_config_identities_are_refused(tmp_path: Path) -> None:
    """Critical 1, axis 2. Folds trained under different backbones, hyperparameters, or
    components are different models that happen to share a split family and window spec --
    not folds of one experiment. The reviewer's real-run example pooled three folds with
    `hidden` 8/16/4, `depth` 1/2/1, different `lr`, and one fold missing `transform`
    entirely, without complaint."""
    a = write_fold(tmp_path / "a", 0, [0, 1], config_identity="c" * 64)
    b = write_fold(tmp_path / "b", 1, [2, 3], config_identity="d" * 64)
    with pytest.raises(EvalError, match="config"):
        pool_folds([a, b], metrics=("accuracy",))


def test_folds_differing_only_in_fold_index_still_pool(tmp_path: Path) -> None:
    """Critical 1, axis 3. The positive case: folds of one real experiment share every
    field except `fold` itself, and must pool without complaint."""
    a = write_fold(tmp_path / "a", 0, [0, 1])
    b = write_fold(tmp_path / "b", 1, [2, 3])
    pooled = pool_folds([a, b], metrics=("accuracy",))
    assert pooled.folds == (0, 1)


def test_pooling_refuses_when_a_declared_fold_never_ran(tmp_path: Path) -> None:
    """Finding 8. `pool_folds` cannot know how many folds an experiment should have on its
    own -- it never opens a split file -- so this is opt-in via `expected_folds`, which a
    caller fills in from the `SplitFile` it already loaded."""
    a = write_fold(tmp_path / "a", 0, [0, 1])
    b = write_fold(tmp_path / "b", 1, [2, 3])
    with pytest.raises(EvalError, match="fold"):
        pool_folds([a, b], metrics=("accuracy",), expected_folds=[0, 1, 2])


def test_pooling_every_declared_fold_is_fine(tmp_path: Path) -> None:
    a = write_fold(tmp_path / "a", 0, [0, 1])
    b = write_fold(tmp_path / "b", 1, [2, 3])
    pooled = pool_folds([a, b], metrics=("accuracy",), expected_folds=[0, 1])
    assert pooled.folds == (0, 1)


def test_a_fold_that_never_ran_is_named(tmp_path: Path) -> None:
    a = write_fold(tmp_path / "a", 0, [0, 1])
    with pytest.raises(EvalError, match="does not exist"):
        pool_folds([a, tmp_path / "missing"], metrics=("accuracy",))


def test_pooling_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="at least one fold"):
        pool_folds([], metrics=("accuracy",))


def _write_via_the_real_writer(
    directory: Path,
    fold_index: int,
    rows: list[int],
    *,
    y_score: np.ndarray | None,
) -> Path:
    """Build one fold's `predictions.npz` through the actual writer, not a hand-rolled
    `np.savez_compressed` -- `write_fold` above stands in for the writer in every other test
    here, which is the right isolation for testing `pool_folds`' own guards, but it means no
    test in this file proves the writer and this reader agree on the npz shape, or that the
    writer's own `if result.y_score is not None` branch (both sides of it) is ever taken in
    anything but the writer's own unreachable-in-production else path (findings 5 and 6:
    `_assemble` always sets a score, so this branch never sees `None` via `run_torch`)."""
    from dsio.eval.contract import Fold, FoldPrediction
    from dsio.train.torch_task import _write_predictions

    fold = Fold(index=fold_index, train=np.array([100 + fold_index]), test=np.asarray(rows))
    result = FoldPrediction(
        y_true=np.asarray([r % 2 for r in rows], dtype=np.int64),
        y_pred=np.asarray([r % 2 for r in rows], dtype=np.int64),
        y_score=y_score,
    )
    _write_predictions(
        directory,
        fold=fold,
        result=result,
        split="fam",
        split_digest=DIGEST,
        window_digest=WINDOW_DIGEST,
        config_identity=CONFIG_IDENTITY,
    )
    return directory


def test_scores_survive_pooling_exactly_not_approximately(tmp_path: Path) -> None:
    """A threshold sweep on rounded scores finds the wrong one.

    Moved here from the deleted `test_loop.py`, which proved the same property of
    `OutOfFold.save`/`.load` directly. Under fold-as-process the npz round trip happens in
    two places instead of one -- the writer in `torch_task._write_predictions` and the
    reader here in `pool_folds` -- so this is where it is re-proven now that both exist.

    Goes through `_write_predictions` itself (finding 6): a prior version of this test built
    its npz with a bare `np.savez_compressed`, which proved `pool_folds` preserves whatever
    dtype it is handed but never proved the writer hands it float64 in the first place. A
    writer that narrowed `y_score` to float32 before scoring -- exactly the precision loss
    this test exists to catch -- passed every assertion here until the writer was actually
    exercised.
    """
    rng = np.random.default_rng(0)
    scores = rng.random(50)
    a = _write_via_the_real_writer(tmp_path / "a", 0, list(range(50)), y_score=scores)
    pooled = pool_folds([a], metrics=("accuracy",))
    assert pooled.y_score is not None
    assert pooled.y_score.dtype == np.float64
    assert np.array_equal(pooled.y_score, scores)


def test_pooling_a_writer_produced_mixture_of_scored_and_unscored_folds_is_rejected(
    tmp_path: Path,
) -> None:
    """Finding 5. Guard 3's `y_score is None` branch never fires via `run_torch` in
    production (`_assemble` always returns a score), so nothing proved the writer's own
    `if result.y_score is not None:` actually omits the key on the untaken branch, rather
    than e.g. writing zeros -- the reviewer's mutation (`arrays["y_score"] =
    np.zeros_like(...)`) left every other test in this file green. This builds both files
    through `_write_predictions` itself, one with a score and one without, and proves the
    guard still fires."""
    scored = _write_via_the_real_writer(
        tmp_path / "a", 0, [0, 1], y_score=np.asarray([0.9, 0.1])
    )
    unscored = _write_via_the_real_writer(tmp_path / "b", 1, [2, 3], y_score=None)
    with pytest.raises(EvalError, match="mixture of"):
        pool_folds([scored, unscored], metrics=("accuracy",))
