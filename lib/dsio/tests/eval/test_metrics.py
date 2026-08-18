"""Metric correctness, pinned against scikit-learn.

dsio implements these in numpy so the artifact contract does not depend on an optional
extra. That is only defensible if "ours" and "sklearn's" are the same number, so this file
proves it — including the tie-heavy and degenerate cases where hand-rolled implementations
usually diverge.

sklearn is a dev dependency, present in CI but not required to *use* dsio. These tests are
the bridge between those two facts.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsio.eval.metrics import (
    METRICS,
    MetricError,
    accuracy,
    average_precision,
    balanced_accuracy,
    compute,
    f1_macro,
    log_loss,
    mae,
    positive_rate,
    r2,
    rmse,
    roc_auc,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics")

TOL = 1e-12


@pytest.fixture
def binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    truth = rng.binomial(1, 0.3, size=500)
    score = np.clip(truth * 0.35 + rng.normal(0.3, 0.25, size=500), 0.001, 0.999)
    return truth, score


@pytest.fixture
def tied() -> tuple[np.ndarray, np.ndarray]:
    """Heavy ties: a coarse tree ensemble, or any model on a small fold, produces these.

    Ranking metrics that ignore ties are optimistically biased, and the bias grows exactly
    where evaluation sets are small — which is where the estimate is already fragile.
    """
    rng = np.random.default_rng(1)
    truth = rng.binomial(1, 0.4, size=300)
    score = np.round(rng.uniform(0, 1, size=300), 1)
    return truth, score


# --- parity with scikit-learn -------------------------------------------------------


def test_average_precision_matches_sklearn(binary) -> None:
    truth, score = binary
    assert average_precision(truth, score > 0.5, score) == pytest.approx(
        sklearn_metrics.average_precision_score(truth, score), abs=TOL
    )


def test_average_precision_matches_sklearn_with_heavy_ties(tied) -> None:
    truth, score = tied
    assert average_precision(truth, score > 0.5, score) == pytest.approx(
        sklearn_metrics.average_precision_score(truth, score), abs=TOL
    )


def test_average_precision_is_not_the_trapezoid(tied) -> None:
    """The specific mistake this implementation exists to avoid.

    `auc(recall, precision)` interpolates between operating points, reporting performance
    at thresholds the model cannot achieve. It is a common line in Kaggle notebooks and it
    is optimistically biased.
    """
    truth, score = tied
    precision, recall, _ = sklearn_metrics.precision_recall_curve(truth, score)
    trapezoid = sklearn_metrics.auc(recall, precision)
    step_wise = average_precision(truth, score > 0.5, score)
    assert step_wise != pytest.approx(trapezoid, abs=1e-6)
    assert step_wise == pytest.approx(
        sklearn_metrics.average_precision_score(truth, score), abs=TOL
    )


def test_roc_auc_matches_sklearn(binary) -> None:
    truth, score = binary
    assert roc_auc(truth, score > 0.5, score) == pytest.approx(
        sklearn_metrics.roc_auc_score(truth, score), abs=TOL
    )


def test_roc_auc_matches_sklearn_with_heavy_ties(tied) -> None:
    truth, score = tied
    assert roc_auc(truth, score > 0.5, score) == pytest.approx(
        sklearn_metrics.roc_auc_score(truth, score), abs=TOL
    )


def test_roc_auc_of_a_constant_score_is_one_half(tied) -> None:
    """Every value tied. A rank implementation that drops ties reports 1.0 here."""
    truth, _ = tied
    assert roc_auc(truth, truth, np.full(truth.size, 0.7)) == pytest.approx(0.5, abs=TOL)


def test_f1_macro_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 4, size=400)
    predicted = np.where(rng.random(400) < 0.7, truth, rng.integers(0, 4, size=400))
    assert f1_macro(truth, predicted) == pytest.approx(
        sklearn_metrics.f1_score(truth, predicted, average="macro"), abs=TOL
    )


def test_balanced_accuracy_matches_sklearn() -> None:
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 3, size=300)
    predicted = np.where(rng.random(300) < 0.6, truth, rng.integers(0, 3, size=300))
    assert balanced_accuracy(truth, predicted) == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(truth, predicted), abs=TOL
    )


def test_accuracy_matches_sklearn() -> None:
    truth = np.array([0, 1, 1, 0, 1])
    predicted = np.array([0, 1, 0, 0, 1])
    assert accuracy(truth, predicted) == pytest.approx(
        sklearn_metrics.accuracy_score(truth, predicted), abs=TOL
    )


def test_log_loss_matches_sklearn(binary) -> None:
    truth, score = binary
    assert log_loss(truth, score > 0.5, score) == pytest.approx(
        sklearn_metrics.log_loss(truth, score), abs=1e-10
    )


def test_regression_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(4)
    truth = rng.normal(size=200)
    predicted = truth + rng.normal(scale=0.3, size=200)
    assert rmse(truth, predicted) == pytest.approx(
        float(np.sqrt(sklearn_metrics.mean_squared_error(truth, predicted))), abs=TOL
    )
    assert mae(truth, predicted) == pytest.approx(
        sklearn_metrics.mean_absolute_error(truth, predicted), abs=TOL
    )
    assert r2(truth, predicted) == pytest.approx(
        sklearn_metrics.r2_score(truth, predicted), abs=TOL
    )


# --- refusals -----------------------------------------------------------------------


def test_ranking_metrics_refuse_hard_labels() -> None:
    """Scoring thresholded labels as if they were probabilities is silent nonsense."""
    truth = np.array([0, 1, 1, 0])
    with pytest.raises(MetricError, match="continuous scores"):
        average_precision(truth, truth, None)
    with pytest.raises(MetricError, match="continuous scores"):
        roc_auc(truth, truth, None)


def test_average_precision_refuses_a_fold_with_no_positives() -> None:
    """Silently returning 0.0 or NaN would average into the headline unnoticed."""
    truth = np.zeros(20, dtype=int)
    with pytest.raises(MetricError, match="no positive labels"):
        average_precision(truth, truth, np.linspace(0, 1, 20))


def test_roc_auc_refuses_a_single_class() -> None:
    truth = np.ones(20, dtype=int)
    with pytest.raises(MetricError, match="one class is absent"):
        roc_auc(truth, truth, np.linspace(0, 1, 20))


def test_binary_metrics_refuse_multiclass() -> None:
    truth = np.array([0, 1, 2, 1, 0])
    with pytest.raises(MetricError, match="binary target"):
        average_precision(truth, truth, np.linspace(0, 1, 5))


def test_r2_refuses_a_constant_target() -> None:
    with pytest.raises(MetricError, match="undefined"):
        r2(np.full(10, 3.0), np.arange(10, dtype=float))


# --- the registry -------------------------------------------------------------------


def test_compute_resolves_every_name_before_computing_any() -> None:
    """A typo in the fifth metric must not cost the four expensive ones first."""
    calls: list[str] = []

    @METRICS.register("_counting_probe")
    def _probe(y_true, y_pred, y_score=None):  # type: ignore[no-untyped-def]
        calls.append("ran")
        return 1.0

    truth = np.array([0, 1, 1, 0])
    with pytest.raises(KeyError, match="unknown metric"):
        compute(["_counting_probe", "nope"], truth, truth)
    assert calls == []


def test_unknown_metric_suggests_a_close_name() -> None:
    truth = np.array([0, 1])
    with pytest.raises(KeyError, match="did you mean"):
        compute(["accuarcy"], truth, truth)


def test_positive_rate_is_the_floor_average_precision_sits_on() -> None:
    """An AP of 0.30 is excellent at a 2% base rate and worthless at 0.29."""
    truth = np.concatenate([np.ones(2), np.zeros(98)])
    assert positive_rate(truth, truth) == pytest.approx(0.02)
