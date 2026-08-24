"""Verdict invariants: the noise floor, the paired test, and the comparison it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from dsio.eval.contract import EvalError, Fold, fold_fingerprint
from dsio.eval.pool import Pooled
from dsio.eval.verdict import (
    Outcome,
    compare,
    compare_all,
    minimum_detectable_rows,
    noise_floor,
    paired_noise_floor,
    sampling_noise,
    verdict,
)


def pooled(
    scores: list[float],
    *,
    pooled_value: float | None = None,
    metric: str = "accuracy",
    split: str = "fam",
    split_digest: str = "digest-a",
    folds: tuple[int, ...] | None = None,
) -> Pooled:
    """A :class:`Pooled` shaped the way :func:`dsio.eval.pool.pool_folds` would produce one,
    with per-fold scores set directly rather than computed from raw predictions -- what
    `compare` is judged on is the fold-metric series, not the arithmetic that produced it.
    """
    fold_ids = folds if folds is not None else tuple(range(len(scores)))
    n = len(scores)
    return Pooled(
        row_id=np.arange(n, dtype=np.int64),
        fold=np.asarray(fold_ids, dtype=np.int64),
        y_true=np.zeros(n, dtype=np.int64),
        y_pred=np.zeros(n, dtype=np.int64),
        y_score=None,
        metrics={metric: pooled_value if pooled_value is not None else float(np.mean(scores))},
        fold_metrics={index: {metric: value} for index, value in zip(fold_ids, scores)},
        folds=fold_ids,
        split=split,
        split_digest=split_digest,
    )


# --- the hand-written scripts-compatible verdict --------------------------------------------------


def test_an_improvement_below_the_fold_spread_is_neutral() -> None:
    """The whole point. A 0.001 gain against a 0.02 fold spread is variance."""
    result = verdict(0.851, 0.850, [0.83, 0.87, 0.84, 0.88], higher_is_better=True)
    assert result.outcome is Outcome.NEUTRAL


def test_an_improvement_above_the_fold_spread_is_a_win() -> None:
    result = verdict(0.95, 0.85, [0.84, 0.86, 0.85, 0.85], higher_is_better=True)
    assert result.outcome is Outcome.WIN


def test_a_drop_beyond_the_floor_is_a_regression() -> None:
    result = verdict(0.70, 0.85, [0.84, 0.86, 0.85, 0.85], higher_is_better=True)
    assert result.outcome is Outcome.REGRESSION


def test_direction_is_honoured_for_error_metrics() -> None:
    """Read an RMSE as higher-is-better and every verdict in the table inverts."""
    lower = verdict(0.20, 0.40, [0.39, 0.41], higher_is_better=False)
    assert lower.outcome is Outcome.WIN
    assert verdict(0.20, 0.40, [0.39, 0.41], higher_is_better=True).outcome is Outcome.REGRESSION


def test_a_missing_score_is_unknown_not_zero() -> None:
    assert verdict(None, 0.85).outcome is Outcome.UNKNOWN
    assert verdict(0.85, float("nan")).outcome is Outcome.UNKNOWN


def test_one_fold_gives_no_floor() -> None:
    """A single fold carries no information about spread; a fabricated floor would hide."""
    assert noise_floor([0.85]) == 0.0
    assert noise_floor([]) == 0.0


# --- the paired test ------------------------------------------------------------------


def test_paired_comparison_sees_what_the_unpaired_floor_buries() -> None:
    """The improvement dsio's committed folds buy.

    Both models are hurt by the same hard fold. Unpaired, that shared difficulty inflates
    the floor above a real and perfectly consistent improvement; paired, it cancels. This is
    the whole reason pairing exists -- the property must keep holding.
    """
    baseline_scores = [0.70, 0.90, 0.75, 0.95]
    candidate_scores = [s + 0.01 for s in baseline_scores]

    unpaired = noise_floor(candidate_scores)
    paired = paired_noise_floor(candidate_scores, baseline_scores)
    assert paired < 1e-9 < unpaired

    result = compare(pooled(candidate_scores), pooled(baseline_scores), metric="accuracy")
    assert result.method == "paired"
    assert result.outcome is Outcome.WIN
    assert result.improvement == pytest.approx(0.01)


def test_a_consistent_improvement_stays_neutral_when_folds_cannot_be_paired() -> None:
    """The same numbers, judged without the pairing guarantee, are correctly not a win."""
    baseline_scores = [0.70, 0.90, 0.75, 0.95]
    candidate_scores = [s + 0.01 for s in baseline_scores]
    result = compare(
        pooled(candidate_scores, split_digest="digest-one"),
        pooled(baseline_scores, split_digest="digest-two"),
        metric="accuracy",
        require_same_folds=False,
    )
    assert result.method == "unpaired"
    assert result.outcome is Outcome.NEUTRAL


def test_paired_noise_floor_rejects_mismatched_fold_counts() -> None:
    with pytest.raises(EvalError, match="equal fold counts"):
        paired_noise_floor([0.1, 0.2, 0.3], [0.1, 0.2])


def test_an_inconsistent_improvement_is_not_a_win_even_when_paired() -> None:
    """Pairing sharpens the test; it does not lower the bar. A gain that appears in two
    folds and reverses in the other two is still noise."""
    baseline_scores = [0.80, 0.80, 0.80, 0.80]
    candidate_scores = [0.90, 0.70, 0.90, 0.70]
    result = compare(pooled(candidate_scores), pooled(baseline_scores), metric="accuracy")
    assert result.method == "paired"
    assert result.outcome is Outcome.NEUTRAL


# --- the refusal ----------------------------------------------------------------------


def test_comparing_different_fold_assignments_is_refused() -> None:
    """The doctrine is easy to state and impossible to enforce without a correspondence
    check. A refusal beats a confident, meaningless delta."""
    with pytest.raises(EvalError, match="different folds"):
        compare(
            pooled([0.9, 0.9], folds=(0, 1)),
            pooled([0.8, 0.8], folds=(2, 3)),
            metric="accuracy",
        )


def test_comparing_fold_2_of_one_split_against_fold_2_of_another_is_refused() -> None:
    """The gain decision 6 buys over the old whole-report fingerprint: fold-as-process
    means a comparison can be attempted one fold at a time, and matching on split digest
    plus fold index catches it even though both sides happen to name the same fold index."""
    with pytest.raises(EvalError, match="store snapshots"):
        compare(
            pooled([0.9], folds=(2,), split="fam", split_digest="digest-a"),
            pooled([0.8], folds=(2,), split="fam", split_digest="digest-b"),
            metric="accuracy",
        )


def test_comparing_different_split_families_is_refused_first() -> None:
    with pytest.raises(EvalError, match="different split families"):
        compare(
            pooled([0.9, 0.9], split="family-a"),
            pooled([0.8, 0.8], split="family-b"),
            metric="accuracy",
        )


def test_the_refusal_can_be_waived_deliberately() -> None:
    result = compare(
        pooled([0.9, 0.9], folds=(0, 1)),
        pooled([0.8, 0.8], folds=(2, 3)),
        metric="accuracy",
        require_same_folds=False,
    )
    assert result.outcome is Outcome.WIN
    assert "folds differ" in (result.reason or "")


def test_a_missing_metric_is_unknown_and_says_what_was_recorded() -> None:
    result = compare(pooled([0.9, 0.9]), pooled([0.8, 0.8]), metric="roc_auc")
    assert result.outcome is Outcome.UNKNOWN
    assert "accuracy" in (result.reason or "")


def test_compare_all_requires_an_explicit_direction_per_metric() -> None:
    with pytest.raises(EvalError, match="no such metric"):
        compare_all(pooled([0.9, 0.9]), pooled([0.8, 0.8]), directions={"rmse": False})


def test_compare_all_returns_one_row_per_metric() -> None:
    candidate = pooled([0.9, 0.92])
    baseline = pooled([0.8, 0.82])
    rows = compare_all(candidate, baseline, directions={"accuracy": True})
    assert [row.metric for row in rows] == ["accuracy"]
    assert rows[0].outcome is Outcome.WIN


# --- the fingerprint (still used to hash an exact fold assignment; unrelated to `compare`,
# which now matches on split digest and fold-index correspondence instead) -----------------


def test_identical_fold_assignments_fingerprint_identically() -> None:
    left = [Fold(index=0, train=np.arange(10, 20), test=np.arange(0, 10))]
    right = [Fold(index=0, train=np.arange(10, 30), test=np.arange(0, 10))]
    assert fold_fingerprint(left) == fold_fingerprint(right), "only held-out rows matter"


def test_a_different_held_out_set_fingerprints_differently() -> None:
    left = [Fold(index=0, train=np.arange(10, 20), test=np.arange(0, 10))]
    right = [Fold(index=0, train=np.arange(11, 20), test=np.arange(1, 11))]
    assert fold_fingerprint(left) != fold_fingerprint(right)


def test_fingerprint_ignores_the_order_folds_were_listed_in() -> None:
    a = Fold(index=0, train=np.arange(10, 20), test=np.arange(0, 10))
    b = Fold(index=1, train=np.arange(0, 10), test=np.arange(10, 20))
    assert fold_fingerprint([a, b]) == fold_fingerprint([b, a])


# --- the second, independent floor ----------------------------------------------------


def test_sampling_noise_shrinks_with_the_evaluation_set() -> None:
    assert sampling_noise(100) > sampling_noise(10_000)
    assert sampling_noise(10_000) == pytest.approx(0.005)


def test_sampling_noise_rejects_impossible_inputs() -> None:
    with pytest.raises(EvalError, match="must be positive"):
        sampling_noise(0)
    with pytest.raises(EvalError, match="proportion"):
        sampling_noise(100, p=1.5)


def test_minimum_detectable_rows_answers_the_question_worth_asking_first() -> None:
    """Resolving a 0.001 accuracy difference takes a quarter-million evaluation rows."""
    assert minimum_detectable_rows(0.001) == 250_000
    with pytest.raises(EvalError, match="must be positive"):
        minimum_detectable_rows(0.0)


def test_a_comparison_reports_its_improvement_as_a_multiple_of_the_floor() -> None:
    result = compare(pooled([0.90, 0.93, 0.88]), pooled([0.80, 0.84, 0.80]), metric="accuracy")
    assert result.noise_floor is not None and result.noise_floor > 0
    assert result.ratio is not None and result.ratio > 1.0
    assert "WIN" in result.summary_line()


def test_a_perfectly_consistent_difference_has_no_floor_left_to_clear() -> None:
    """The zero-floor edge, pinned as deliberate rather than discovered later.

    When every fold shows exactly the same difference there is no fold-to-fold variance to
    subtract, so the paired floor is genuinely 0 and any improvement is a win. That is the
    correct reading of the evidence — but it is also why `sampling_noise` exists as a
    second, independent floor: consistency across folds says nothing about whether the
    evaluation set was large enough to resolve the difference at all.
    """
    result = compare(pooled([0.90, 0.91]), pooled([0.80, 0.81]), metric="accuracy")
    assert result.noise_floor == 0.0
    assert result.outcome is Outcome.WIN
    assert result.ratio is None
