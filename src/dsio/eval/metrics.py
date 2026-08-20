"""Metrics, implemented in numpy and registered by name.

Two reasons these are not simply re-exported from scikit-learn.

**Dependency.** sklearn is an optional extra. If the metric path needed it, the fixed
artifact contract — the thing every runner writes and every comparison reads — would be
untestable in a bare install, and a torch-only or forecast-only project would drag in
sklearn to compute an RMSE.

**Control over the one that matters.** Average precision is dsio's headline metric for
imbalanced problems, and it is the metric people most often compute wrongly: the trapezoid
interpolation used by ``auc(recall, precision)`` is optimistically biased. The step-wise
sum here is the correct estimator, and ``tests/eval/test_metrics.py`` pins it against
scikit-learn to 1e-12 so "we wrote our own" never becomes "ours is subtly different".

A metric takes ``(y_true, y_pred, y_score)`` and returns one float. ``y_score`` is the
continuous output — probability of the positive class, or the regression value — and is
``None`` when the runner produced only hard predictions. A metric that needs it says so
rather than silently scoring the thresholded labels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dsio.config.registry import Registry

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]

METRICS: Registry[MetricFn] = Registry("metric")


class MetricError(ValueError):
    """Raised when a metric cannot be computed from what the runner produced."""


def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Register a metric under ``name``."""
    return METRICS.register(name)


def compute(
    names: tuple[str, ...] | list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate every named metric, resolving all names before computing any.

    Resolving first means a typo in the fifth metric fails before the first one runs, which
    matters when a metric is expensive and the run has already spent an hour training.
    """
    functions = [(name, METRICS.get(name)) for name in names]
    return {name: float(function(y_true, y_pred, y_score)) for name, function in functions}


def _require_score(y_score: np.ndarray | None, name: str) -> np.ndarray:
    if y_score is None:
        raise MetricError(
            f"{name} needs continuous scores, but the runner produced only hard "
            f"predictions; return y_score from fit_predict or drop {name} from metrics"
        )
    score = np.asarray(y_score, dtype=np.float64)
    if score.ndim == 2:
        if score.shape[1] != 2:
            raise MetricError(
                f"{name} is binary-only, but y_score has {score.shape[1]} columns; "
                "score one-vs-rest yourself, or use a multiclass metric"
            )
        return score[:, 1]
    return score


def _binary(y_true: np.ndarray) -> np.ndarray:
    labels = np.unique(y_true)
    if labels.size > 2:
        raise MetricError(
            f"expected a binary target, found {labels.size} classes; "
            "use accuracy, f1_macro or balanced_accuracy for multiclass"
        )
    return (np.asarray(y_true) > 0).astype(np.int64)


# --- classification -----------------------------------------------------------------


@metric("accuracy")
def accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@metric("balanced_accuracy")
def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> float:
    """Mean per-class recall. The honest headline when classes are imbalanced."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [
        float(np.mean(y_pred[y_true == label] == label))
        for label in np.unique(y_true)
        if np.any(y_true == label)
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, label: object) -> float:
    tp = float(np.sum((y_true == label) & (y_pred == label)))
    fp = float(np.sum((y_true != label) & (y_pred == label)))
    fn = float(np.sum((y_true == label) & (y_pred != label)))
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else 2.0 * tp / denominator


@metric("f1_macro")
def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    """Unweighted mean F1 over every class present in truth or prediction.

    Including predicted-but-absent classes is deliberate: a model that invents a class
    should be penalised for it, not have the evidence averaged away.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    return float(np.mean([_f1_per_class(y_true, y_pred, label) for label in labels]))


@metric("f1")
def f1_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    """F1 of the positive class."""
    return _f1_per_class(_binary(y_true), (np.asarray(y_pred) > 0).astype(np.int64), 1)


@metric("precision")
def precision(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> float:
    truth, predicted = _binary(y_true), (np.asarray(y_pred) > 0).astype(np.int64)
    hits = float(np.sum((truth == 1) & (predicted == 1)))
    denominator = float(np.sum(predicted == 1))
    return 0.0 if denominator == 0.0 else hits / denominator


@metric("recall")
def recall(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    truth, predicted = _binary(y_true), (np.asarray(y_pred) > 0).astype(np.int64)
    hits = float(np.sum((truth == 1) & (predicted == 1)))
    denominator = float(np.sum(truth == 1))
    return 0.0 if denominator == 0.0 else hits / denominator


@metric("average_precision")
def average_precision(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> float:
    """Area under the precision-recall curve, summed step-wise.

    ``sum_n (R_n - R_{n-1}) * P_n`` over distinct score thresholds — no interpolation
    between operating points, because interpolating a PR curve reports performance at
    thresholds the model cannot actually achieve.
    """
    truth = _binary(y_true)
    score = _require_score(y_score, "average_precision")
    positives = int(np.sum(truth))
    if positives == 0:
        raise MetricError("average_precision is undefined with no positive labels")

    order = np.argsort(-score, kind="mergesort")
    truth, score = truth[order], score[order]
    # One operating point per distinct score; ties must share a threshold or precision
    # jumps at boundaries the model cannot distinguish.
    boundaries = np.flatnonzero(np.diff(score))
    thresholds = np.concatenate([boundaries, [truth.size - 1]])

    true_positives = np.cumsum(truth)[thresholds]
    predicted_positives = thresholds + 1
    precisions = true_positives / predicted_positives
    recalls = true_positives / positives
    increments = np.diff(np.concatenate([[0.0], recalls]))
    return float(np.sum(increments * precisions))


@metric("roc_auc")
def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    """Rank-based AUC (Mann-Whitney U), with ties given their average rank.

    Ties are not a corner case: a model that outputs a constant, or a tree ensemble on a
    small validation fold, produces many. Ignoring them inflates AUC.
    """
    truth = _binary(y_true)
    score = _require_score(y_score, "roc_auc")
    n_pos = int(np.sum(truth))
    n_neg = truth.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise MetricError("roc_auc is undefined when one class is absent")
    ranks = _average_ranks(score)
    return float((np.sum(ranks[truth == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, tied values sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    per_group = first + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(per_group, counts)
    return ranks


@metric("log_loss")
def log_loss(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    truth = _binary(y_true)
    score = np.clip(_require_score(y_score, "log_loss"), 1e-15, 1 - 1e-15)
    return float(-np.mean(truth * np.log(score) + (1 - truth) * np.log(1 - score)))


@metric("positive_rate")
def positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> float:
    """Base rate of the target. Logged so an AP is never read without its floor.

    An average precision of 0.30 is excellent at a 2% base rate and worthless at 0.29.
    Recording the two apart from each other is how that mistake gets made.
    """
    return float(np.mean(_binary(y_true)))


# --- regression ---------------------------------------------------------------------


@metric("rmse")
def rmse(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(residual**2)))


@metric("mae")
def mae(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(residual)))


@metric("r2")
def r2(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total == 0.0:
        raise MetricError("r2 is undefined when the target is constant")
    residual = float(np.sum((truth - np.asarray(y_pred, dtype=np.float64)) ** 2))
    return 1.0 - residual / total


@metric("smape")
def smape(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> float:
    """Symmetric MAPE, the standard forecasting error. Bounded, so a near-zero actual
    cannot make one horizon dominate the whole score."""
    truth = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    denominator = np.abs(truth) + np.abs(predicted)
    ratio = np.where(denominator == 0.0, 0.0, np.abs(truth - predicted) / np.where(
        denominator == 0.0, 1.0, denominator
    ))
    return float(2.0 * np.mean(ratio))
