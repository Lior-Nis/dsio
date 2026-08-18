"""The fold loop: framework code, not user code.

``for fold: fit -> predict -> accumulate -> score -> log``. Four scripts in `kaggler`
re-implement this by hand, and each one gets a slightly different definition of what "the
score" means — averaged per fold in one, pooled in another. Writing it once is the point.

The loop knows nothing about models, data or frameworks. It receives folds as row
positions and a ``fit_predict`` callable, which is what lets the same code drive an sklearn
pipeline, a Lightning trainer and a Nixtla rolling origin without adapters.

Three things it refuses to let pass:

- **predictions that do not line up with the fold they came from** — a silent off-by-one in
  a runner would otherwise score row *i*'s prediction against row *j*'s label and produce a
  number that looks merely disappointing rather than wrong;
- **a row predicted by two folds** — either the folds overlap or a runner returned the
  wrong positions, and both mean the pooled metric double-counts;
- **scores present for some folds and absent for others** — pooling those produces a
  ranking metric computed on a mixture of probabilities and hard labels.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from dsio.eval.contract import (
    CVReport,
    EvalError,
    Fold,
    FoldMetrics,
    FoldPrediction,
    OutOfFold,
    fold_fingerprint,
)
from dsio.eval.metrics import MetricError, compute

FitPredict = Callable[[Fold], FoldPrediction]


def cross_validate(
    folds: Sequence[Fold],
    fit_predict: FitPredict,
    *,
    metrics: Sequence[str],
    n_rows: int,
    on_fold: Callable[[FoldMetrics], None] | None = None,
) -> tuple[CVReport, OutOfFold]:
    """Run every fold, accumulate out-of-fold predictions, and score them.

    ``on_fold`` is called with each fold's metrics as soon as they exist, so a long run
    streams results into the ledger instead of producing everything at the end. The loop
    itself imports nothing from :mod:`dsio.runs`; the caller wires that up.
    """
    if not folds:
        raise EvalError("cross_validate needs at least one fold")
    names = list(metrics)

    row_ids: list[np.ndarray] = []
    fold_ids: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    fold_metrics: list[FoldMetrics] = []
    scored: list[str] = []

    seen: set[int] = set()

    for fold in folds:
        started = time.perf_counter()
        result = fit_predict(fold)
        elapsed = time.perf_counter() - started

        if len(result.y_true) != fold.test.size:
            raise EvalError(
                f"{fold.name}: fit_predict returned {len(result.y_true)} predictions for "
                f"{fold.test.size} test rows; the runner and the fold disagree about what "
                "was held out"
            )

        overlap = seen & set(fold.test.tolist())
        if overlap:
            raise EvalError(
                f"{fold.name}: {len(overlap)} row(s) were already predicted by an earlier "
                f"fold, first at position {min(overlap)}; folds must be disjoint or the "
                "pooled metric double-counts them"
            )
        seen |= set(fold.test.tolist())

        row_ids.append(fold.test)
        fold_ids.append(np.full(fold.test.size, fold.index, dtype=np.int64))
        truths.append(np.asarray(result.y_true))
        predictions.append(np.asarray(result.y_pred))
        if result.y_score is not None:
            scores.append(np.asarray(result.y_score))
            scored.append(fold.name)

        try:
            values = compute(names, result.y_true, result.y_pred, result.y_score)
        except MetricError as error:
            raise EvalError(
                f"{fold.name}: {error}. A fold that cannot be scored is a split problem "
                "before it is a metric problem — check stratification."
            ) from error

        entry = FoldMetrics(
            fold=fold.index,
            name=fold.name,
            sizes=fold.sizes,
            seconds=round(elapsed, 4),
            metrics=values,
        )
        fold_metrics.append(entry)
        if on_fold is not None:
            on_fold(entry)

    if scores and len(scores) != len(folds):
        missing = [fold.name for fold in folds if fold.name not in scored]
        raise EvalError(
            f"{len(scores)} of {len(folds)} folds returned y_score; "
            f"{', '.join(missing[:5])} did not. Pooling a mixture of probabilities and "
            "hard labels produces a ranking metric that means nothing."
        )

    oof = OutOfFold(
        row_id=np.concatenate(row_ids),
        fold=np.concatenate(fold_ids),
        y_true=np.concatenate(truths),
        y_pred=np.concatenate(predictions),
        y_score=np.concatenate(scores) if scores else None,
    )

    pooled = compute(names, oof.y_true, oof.y_pred, oof.y_score)
    return (
        CVReport(
            n_folds=len(folds),
            n_rows=n_rows,
            predicted_rows=len(oof),
            metrics=pooled,
            per_fold_mean=_aggregate(fold_metrics, np.mean),
            per_fold_std=_spread(fold_metrics),
            folds=tuple(fold_metrics),
            scored_folds=tuple(scored),
            has_scores=oof.y_score is not None,
            fold_fingerprint=fold_fingerprint(folds),
        ),
        oof,
    )


def _aggregate(
    folds: Sequence[FoldMetrics], reduce: Callable[[np.ndarray], np.floating]
) -> dict[str, float]:
    if not folds:
        return {}
    return {
        name: float(reduce(np.array([fold.metrics[name] for fold in folds], dtype=np.float64)))
        for name in folds[0].metrics
    }


def _spread(folds: Sequence[FoldMetrics]) -> dict[str, float]:
    """Sample standard deviation across folds, or nothing at all with one fold.

    Reporting 0.0 for a single fold would read as "this result has no uncertainty", which
    is the opposite of true. An absent noise floor is honest; a zero one is a lie the
    verdict machinery would then act on.
    """
    if len(folds) < 2:
        return {}
    return {
        name: float(np.std([fold.metrics[name] for fold in folds], ddof=1))
        for name in folds[0].metrics
    }
