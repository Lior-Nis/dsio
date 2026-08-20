"""Fold loop invariants: what it refuses, and what it records that a metric alone cannot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.eval.contract import (
    EvalError,
    Fold,
    FoldPrediction,
    OutOfFold,
    read_report,
    write_report,
)
from dsio.eval.loop import cross_validate


def linear_folds(n_rows: int, k: int) -> list[Fold]:
    blocks = np.array_split(np.arange(n_rows), k)
    return [
        Fold(index=i, train=np.setdiff1d(np.arange(n_rows), block), test=block)
        for i, block in enumerate(blocks)
    ]


@pytest.fixture
def truth() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.binomial(1, 0.4, size=200)


def perfect(truth: np.ndarray):  # type: ignore[no-untyped-def]
    def fit_predict(fold: Fold) -> FoldPrediction:
        held = truth[fold.test]
        return FoldPrediction(y_true=held, y_pred=held, y_score=held.astype(float))

    return fit_predict


# --- what a Fold refuses ------------------------------------------------------------


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


# --- what the loop refuses ----------------------------------------------------------


def test_misaligned_predictions_are_rejected(truth: np.ndarray) -> None:
    """An off-by-one in a runner would otherwise score row i against row j's label.

    That produces a number which looks merely disappointing, not wrong, so nobody
    investigates it.
    """

    def short(fold: Fold) -> FoldPrediction:
        held = truth[fold.test][:-1]
        return FoldPrediction(y_true=held, y_pred=held)

    with pytest.raises(EvalError, match="disagree about what was held out"):
        cross_validate(linear_folds(200, 4), short, metrics=["accuracy"], n_rows=200)


def test_a_row_predicted_by_two_folds_is_rejected(truth: np.ndarray) -> None:
    """Double-counting quietly reweights the pooled metric toward the duplicated rows.

    Each fold here is individually valid — its own train and test are disjoint — so this
    can only be caught by looking across folds, which is the loop's job and not ``Fold``'s.
    """
    folds = [
        Fold(index=0, train=np.arange(100, 200), test=np.arange(0, 100)),
        Fold(index=1, train=np.arange(150, 200), test=np.arange(50, 150)),
    ]
    with pytest.raises(EvalError, match="already predicted by an earlier fold"):
        cross_validate(folds, perfect(truth), metrics=["accuracy"], n_rows=200)


def test_scores_present_for_only_some_folds_are_rejected(truth: np.ndarray) -> None:
    """Pooling probabilities with hard labels produces a ranking metric that means nothing."""

    def sometimes(fold: Fold) -> FoldPrediction:
        held = truth[fold.test]
        score = held.astype(float) if fold.index == 0 else None
        return FoldPrediction(y_true=held, y_pred=held, y_score=score)

    with pytest.raises(EvalError, match="of 4 folds returned y_score"):
        cross_validate(linear_folds(200, 4), sometimes, metrics=["accuracy"], n_rows=200)


def test_an_unscoreable_fold_is_reported_as_a_split_problem() -> None:
    """A test fold with no positives cannot have an AP, and that is a stratification bug."""
    truth = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])

    def fit_predict(fold: Fold) -> FoldPrediction:
        held = truth[fold.test]
        return FoldPrediction(y_true=held, y_pred=held, y_score=held.astype(float))

    folds = [
        Fold(index=0, train=np.arange(0, 100), test=np.arange(100, 200)),
    ]
    with pytest.raises(EvalError, match="check stratification"):
        cross_validate(folds, fit_predict, metrics=["average_precision"], n_rows=200)


def test_no_folds_is_rejected(truth: np.ndarray) -> None:
    with pytest.raises(EvalError, match="at least one fold"):
        cross_validate([], perfect(truth), metrics=["accuracy"], n_rows=200)


# --- what it records ----------------------------------------------------------------


def test_every_fold_is_scored_and_streamed(truth: np.ndarray) -> None:
    """Long runs must surface results as they happen, not only at the end."""
    seen: list[int] = []
    report, _ = cross_validate(
        linear_folds(200, 4),
        perfect(truth),
        metrics=["accuracy"],
        n_rows=200,
        on_fold=lambda fold: seen.append(fold.fold),
    )
    assert seen == [0, 1, 2, 3]
    assert len(report.folds) == 4
    assert all(fold.sizes["train"] + fold.sizes["test"] == 200 for fold in report.folds)


def test_pooled_and_per_fold_mean_are_both_reported_and_differ() -> None:
    """They answer different questions and disagree; reporting one hides that.

    Folds of unequal size make the pooled metric a weighted average, which is the correct
    estimate — but the per-fold spread is what tells you whether the estimate is stable.
    """
    truth = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
    folds = [
        Fold(index=0, train=np.arange(10, 100), test=np.arange(0, 10)),
        Fold(index=1, train=np.arange(0, 10), test=np.arange(10, 100)),
    ]

    def five_wrong(fold: Fold) -> FoldPrediction:
        """A fixed number of errors, so the small fold is hit far harder in rate terms."""
        held = truth[fold.test]
        predicted = held.copy()
        predicted[:5] ^= 1
        return FoldPrediction(y_true=held, y_pred=predicted)

    report, _ = cross_validate(folds, five_wrong, metrics=["accuracy"], n_rows=100)
    # 10 errors over 100 rows pooled, against the mean of 0.50 and 0.944.
    assert report.metrics["accuracy"] == pytest.approx(0.90)
    assert report.per_fold_mean["accuracy"] == pytest.approx(0.7222, abs=1e-4)
    assert report.per_fold_std["accuracy"] > 0.3


def test_a_single_fold_reports_no_noise_floor(truth: np.ndarray) -> None:
    """Zero would read as 'no uncertainty', which the verdict machinery would act on."""
    folds = [Fold(index=0, train=np.arange(0, 150), test=np.arange(150, 200))]
    report, _ = cross_validate(folds, perfect(truth), metrics=["accuracy"], n_rows=200)
    assert report.per_fold_std == {}
    assert report.per_fold_mean["accuracy"] == 1.0


def test_partial_coverage_is_recorded_not_hidden(truth: np.ndarray) -> None:
    """A purged temporal split legitimately predicts a fraction of rows. Reading a pooled
    metric without knowing it covers 25% of the data is how a result gets over-claimed."""
    folds = [Fold(index=0, train=np.arange(0, 100), test=np.arange(100, 150))]
    report, oof = cross_validate(folds, perfect(truth), metrics=["accuracy"], n_rows=200)
    assert report.predicted_rows == 50
    assert report.coverage == 0.25
    assert len(oof) == 50


def test_out_of_fold_rows_carry_their_origin(truth: np.ndarray) -> None:
    """Predictions without row ids cannot be joined back for error analysis."""
    _, oof = cross_validate(linear_folds(200, 4), perfect(truth), metrics=["accuracy"], n_rows=200)
    assert sorted(oof.row_id.tolist()) == list(range(200))
    assert set(oof.fold.tolist()) == {0, 1, 2, 3}
    assert np.array_equal(oof.y_true[np.argsort(oof.row_id)], truth)


# --- the artifact contract ----------------------------------------------------------


def test_report_round_trips(tmp_path: Path, truth: np.ndarray) -> None:
    report, oof = cross_validate(
        linear_folds(200, 4), perfect(truth), metrics=["accuracy", "f1_macro"], n_rows=200
    )
    write_report(tmp_path, report, oof)

    restored, restored_oof = read_report(tmp_path)
    assert restored.metrics == report.metrics
    assert restored.per_fold_std == report.per_fold_std
    assert len(restored.folds) == 4
    assert restored_oof is not None
    assert np.array_equal(restored_oof.row_id, oof.row_id)
    assert np.array_equal(restored_oof.y_score, oof.y_score)


def test_out_of_fold_predictions_survive_float_round_trip(tmp_path: Path) -> None:
    """Exactly, not approximately: a threshold sweep on rounded scores finds the wrong one."""
    rng = np.random.default_rng(0)
    oof = OutOfFold(
        row_id=np.arange(50),
        fold=np.zeros(50, dtype=np.int64),
        y_true=rng.binomial(1, 0.5, 50),
        y_pred=rng.binomial(1, 0.5, 50),
        y_score=rng.random(50),
    )
    oof.save(tmp_path / "oof.npz")
    assert np.array_equal(OutOfFold.load(tmp_path / "oof.npz").y_score, oof.y_score)


def test_reading_a_missing_report_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="no evaluation report"):
        read_report(tmp_path)
