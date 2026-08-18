"""Verdict invariants: the noise floor, the paired test, and the comparison it refuses."""

from __future__ import annotations

import numpy as np
import pytest

from dsio.eval import (
    CVReport,
    EvalError,
    Fold,
    FoldMetrics,
    Outcome,
    compare,
    compare_all,
    fold_fingerprint,
    minimum_detectable_rows,
    noise_floor,
    paired_noise_floor,
    sampling_noise,
    verdict,
)


def report(
    scores: list[float],
    *,
    pooled: float | None = None,
    metric: str = "accuracy",
    fingerprint: str | None = "abc123",
) -> CVReport:
    return CVReport(
        n_folds=len(scores),
        n_rows=1000,
        predicted_rows=1000,
        metrics={metric: pooled if pooled is not None else float(np.mean(scores))},
        per_fold_mean={metric: float(np.mean(scores))},
        per_fold_std={metric: float(np.std(scores, ddof=1))} if len(scores) > 1 else {},
        folds=tuple(
            FoldMetrics(
                fold=i,
                name=f"fold{i}",
                sizes={"train": 800, "test": 200},
                seconds=1.0,
                metrics={metric: value},
            )
            for i, value in enumerate(scores)
        ),
        fold_fingerprint=fingerprint,
    )


# --- the kaggler-compatible verdict --------------------------------------------------


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
    the floor above a real and perfectly consistent improvement; paired, it cancels.
    """
    baseline_scores = [0.70, 0.90, 0.75, 0.95]
    candidate_scores = [s + 0.01 for s in baseline_scores]

    unpaired = noise_floor(candidate_scores)
    paired = paired_noise_floor(candidate_scores, baseline_scores)
    assert paired < 1e-9 < unpaired

    result = compare(
        report(candidate_scores), report(baseline_scores), metric="accuracy"
    )
    assert result.method == "paired"
    assert result.outcome is Outcome.WIN
    assert result.improvement == pytest.approx(0.01)


def test_a_consistent_improvement_stays_neutral_when_folds_cannot_be_paired() -> None:
    """The same numbers, judged without the pairing guarantee, are correctly not a win."""
    baseline_scores = [0.70, 0.90, 0.75, 0.95]
    candidate_scores = [s + 0.01 for s in baseline_scores]
    result = compare(
        report(candidate_scores, fingerprint="one"),
        report(baseline_scores, fingerprint="two"),
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
    result = compare(report(candidate_scores), report(baseline_scores), metric="accuracy")
    assert result.method == "paired"
    assert result.outcome is Outcome.NEUTRAL


# --- the refusal ----------------------------------------------------------------------


def test_comparing_different_fold_assignments_is_refused() -> None:
    """kaggler states this doctrine in prose and cannot check it. The fingerprint makes
    it enforceable, and a refusal beats a confident, meaningless delta."""
    with pytest.raises(EvalError, match="single source of truth"):
        compare(
            report([0.9, 0.9], fingerprint="one"),
            report([0.8, 0.8], fingerprint="two"),
            metric="accuracy",
        )


def test_the_refusal_can_be_waived_deliberately() -> None:
    result = compare(
        report([0.9, 0.9], fingerprint="one"),
        report([0.8, 0.8], fingerprint="two"),
        metric="accuracy",
        require_same_folds=False,
    )
    assert result.outcome is Outcome.WIN
    assert "folds differ" in (result.reason or "")


def test_a_missing_metric_is_unknown_and_says_what_was_recorded() -> None:
    result = compare(report([0.9, 0.9]), report([0.8, 0.8]), metric="roc_auc")
    assert result.outcome is Outcome.UNKNOWN
    assert "accuracy" in (result.reason or "")


def test_compare_all_requires_an_explicit_direction_per_metric() -> None:
    with pytest.raises(EvalError, match="no such metric"):
        compare_all(report([0.9, 0.9]), report([0.8, 0.8]), directions={"rmse": False})


def test_compare_all_returns_one_row_per_metric() -> None:
    candidate = report([0.9, 0.92])
    baseline = report([0.8, 0.82])
    rows = compare_all(candidate, baseline, directions={"accuracy": True})
    assert [row.metric for row in rows] == ["accuracy"]
    assert rows[0].outcome is Outcome.WIN


# --- the fingerprint ------------------------------------------------------------------


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
    result = compare(report([0.90, 0.93, 0.88]), report([0.80, 0.84, 0.80]), metric="accuracy")
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
    result = compare(report([0.90, 0.91]), report([0.80, 0.81]), metric="accuracy")
    assert result.noise_floor == 0.0
    assert result.outcome is Outcome.WIN
    assert result.ratio is None
