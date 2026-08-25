"""Is this result real? Baseline-relative verdicts, filtered by a noise floor.

**The rule.** An experiment is a win only if its improvement over its baseline exceeds the
noise floor. Otherwise it is neutral, however good the number looks. Chasing improvements
below the floor is how a week disappears into variance.

**The paired floor.** Estimating the floor from the candidate's own fold spread is the
best available when you cannot be sure two runs used the same folds. Decision 6 makes the
process boundary the fold boundary: each run pools N single-fold processes
(:func:`dsio.eval.pool.pool_folds`) and records the split family, the store digest, and
which fold indices it covers. When two pooled runs match on all three, the far sharper
comparison is available: the standard error of the per-fold *differences*. Fold-to-fold
variation the two models share — one fold simply being harder — cancels, and a real 0.002
improvement becomes visible where the unpaired floor would have buried it under a 0.02 fold
spread.

**The refusal.** Two runs that do not correspond — different split families, different
store snapshots, or different folds — are not compared at all. A refusal is worth far more
than a confident, meaningless delta.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import Field

from dsio.contracts import DsioModel
from dsio.eval.contract import EvalError
from dsio.eval.pool import Pooled

DEFAULT_K = 1.0


class Outcome(StrEnum):
    WIN = "win"
    NEUTRAL = "neutral"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class Comparison(DsioModel):
    """One candidate judged against one baseline on one metric."""

    metric: str
    outcome: Outcome
    candidate: float | None = None
    baseline: float | None = None
    improvement: float | None = None
    noise_floor: float | None = None
    method: Literal["paired", "unpaired", "none"] = "none"
    n_folds: int = 0
    higher_is_better: bool = True
    reason: str | None = Field(
        default=None, description="Why the verdict is unknown, or which floor was used."
    )

    @property
    def ratio(self) -> float | None:
        """Improvement in units of the noise floor. Above 1.0 is a win."""
        if self.improvement is None or not self.noise_floor:
            return None
        return self.improvement / self.noise_floor

    def summary_line(self) -> str:
        if self.outcome is Outcome.UNKNOWN:
            return f"{self.metric}: unknown ({self.reason})"
        arrow = "+" if (self.improvement or 0.0) >= 0 else ""
        floor = "" if self.noise_floor is None else f" vs floor {self.noise_floor:.4f}"
        multiple = "" if self.ratio is None else f" ({self.ratio:.1f}x)"
        return (
            f"{self.metric}: {self.outcome.value.upper()} "
            f"{arrow}{self.improvement:.4f}{floor}{multiple} [{self.method}]"
        )


def noise_floor(fold_scores: list[float] | np.ndarray, k: float = DEFAULT_K) -> float:
    """The unpaired floor: ``k`` sample standard deviations across folds.

    Zero with fewer than two folds, because a single fold carries no information about
    spread. Returning zero there is deliberate — it makes every difference a "win", which
    is obviously wrong and therefore gets noticed, where a fabricated floor would not.
    """
    scores = np.asarray(fold_scores, dtype=np.float64)
    if scores.size < 2:
        return 0.0
    return float(k * np.std(scores, ddof=1))


def paired_noise_floor(
    candidate: list[float] | np.ndarray,
    baseline: list[float] | np.ndarray,
    k: float = DEFAULT_K,
) -> float:
    """The paired floor: ``k`` standard errors of the per-fold differences.

    Requires the two runs to have used identical folds. Sharper than the unpaired floor
    because shared fold difficulty cancels: if fold 3 is hard for every model, that
    variance is in both score vectors and vanishes from their difference.

    The standard *error* — sd/sqrt(k) — rather than the standard deviation, because the
    quantity being judged is the mean difference across folds, not one fold's difference.
    """
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.size != right.size:
        raise EvalError(
            f"paired comparison needs equal fold counts, got {left.size} and {right.size}"
        )
    if left.size < 2:
        return 0.0
    differences = left - right
    return float(k * np.std(differences, ddof=1) / math.sqrt(differences.size))


def verdict(
    candidate: float | None,
    baseline: float | None,
    fold_scores: list[float] | np.ndarray | None = None,
    *,
    higher_is_better: bool = True,
    k: float = DEFAULT_K,
    metric: str = "metric",
) -> Comparison:
    """Classify a candidate against a baseline using the unpaired floor.

    The entry point for when all you have is two numbers and a fold
    spread. Prefer :func:`compare`, which takes the pooled runs and can pair the folds.
    """
    if _missing(candidate) or _missing(baseline):
        return Comparison(
            metric=metric,
            outcome=Outcome.UNKNOWN,
            candidate=candidate,
            baseline=baseline,
            higher_is_better=higher_is_better,
            reason="a score is missing",
        )
    assert candidate is not None and baseline is not None
    improvement = (candidate - baseline) if higher_is_better else (baseline - candidate)
    floor = 0.0 if fold_scores is None else noise_floor(fold_scores, k)
    return Comparison(
        metric=metric,
        outcome=_classify(improvement, floor),
        candidate=candidate,
        baseline=baseline,
        improvement=improvement,
        noise_floor=floor,
        method="unpaired" if floor else "none",
        n_folds=0 if fold_scores is None else len(fold_scores),
        higher_is_better=higher_is_better,
        reason=None if floor else "no fold spread available; every difference counts",
    )


def compare(
    candidate: Pooled,
    baseline: Pooled,
    *,
    metric: str,
    higher_is_better: bool = True,
    k: float = DEFAULT_K,
    require_same_folds: bool = True,
) -> Comparison:
    """Judge two pooled, cross-validated runs against each other on one metric.

    Each run is N single-fold processes pooled by :func:`dsio.eval.pool.pool_folds`
    (decision 6: the process boundary is the fold boundary). Pairs the folds when both
    pooled runs share a split family, a store snapshot, and the same set of fold indices —
    the normal case for two runs driven by the same committed split file. Falls back to the
    unpaired floor when that correspondence does not hold and the caller has said that is
    acceptable.
    """
    for pooled, label in ((candidate, "candidate"), (baseline, "baseline")):
        if metric not in pooled.metrics:
            return Comparison(
                metric=metric,
                outcome=Outcome.UNKNOWN,
                higher_is_better=higher_is_better,
                reason=(
                    f"{label} did not record {metric!r}; it has "
                    f"{', '.join(sorted(pooled.metrics)) or 'nothing'}"
                ),
            )

    mismatch = _correspondence(candidate, baseline)
    comparable = mismatch is None
    if require_same_folds and not comparable:
        raise EvalError(
            f"refusing to compare: {mismatch}. Re-run against the same split file, or pass "
            "require_same_folds=False and accept the unpaired floor."
        )

    left = candidate.metrics[metric]
    right = baseline.metrics[metric]
    improvement = (left - right) if higher_is_better else (right - left)

    candidate_folds = _fold_series(candidate, metric)
    baseline_folds = _fold_series(baseline, metric)

    if comparable and candidate_folds is not None and baseline_folds is not None:
        floor = paired_noise_floor(candidate_folds, baseline_folds, k)
        method: Literal["paired", "unpaired", "none"] = "paired"
        reason = "paired across identical folds"
    elif candidate_folds is not None:
        floor = noise_floor(candidate_folds, k)
        method = "unpaired" if floor else "none"
        reason = "folds differ; using the candidate's own spread"
    else:
        floor = 0.0
        method = "none"
        reason = "no per-fold scores; every difference counts"

    return Comparison(
        metric=metric,
        outcome=_classify(improvement, floor),
        candidate=left,
        baseline=right,
        improvement=improvement,
        noise_floor=floor,
        method=method,
        n_folds=len(candidate.folds),
        higher_is_better=higher_is_better,
        reason=reason,
    )


def compare_all(
    candidate: Pooled,
    baseline: Pooled,
    *,
    directions: dict[str, bool],
    k: float = DEFAULT_K,
    require_same_folds: bool = True,
) -> list[Comparison]:
    """Compare every metric named in ``directions``.

    Direction must be stated per metric rather than guessed. An RMSE improvement read as
    higher-is-better inverts every verdict in the table, and nothing about the number
    itself reveals the mistake.
    """
    unknown = set(directions) - set(candidate.metrics)
    if unknown:
        raise EvalError(
            f"no such metric in the candidate report: {', '.join(sorted(unknown))}; "
            f"it recorded {', '.join(sorted(candidate.metrics))}"
        )
    return [
        compare(
            candidate,
            baseline,
            metric=name,
            higher_is_better=higher,
            k=k,
            require_same_folds=require_same_folds,
        )
        for name, higher in sorted(directions.items())
    ]


def sampling_noise(n_rows: int, p: float = 0.5) -> float:
    """Sampling sigma of a proportion metric measured on ``n_rows``.

    Applies to any finite held-out set. A difference below this is invisible in the
    measurement regardless of how many folds agree on it, which is the second, independent
    floor — a 200-row test set cannot resolve a 0.01 accuracy difference no matter how
    stable cross-validation looks.
    """
    if n_rows <= 0:
        raise EvalError("n_rows must be positive")
    if not 0.0 <= p <= 1.0:
        raise EvalError(f"p must be a proportion in [0, 1], got {p}")
    return math.sqrt(p * (1.0 - p) / n_rows)


def minimum_detectable_rows(delta: float, p: float = 0.5) -> int:
    """How many held-out rows are needed before ``delta`` clears the sampling floor.

    The question worth asking *before* running the experiment: at a 0.5 base rate,
    resolving a 0.001 accuracy difference takes a quarter of a million evaluation rows.
    """
    if delta <= 0:
        raise EvalError("delta must be positive")
    return math.ceil(p * (1.0 - p) / (delta * delta))


def _fold_series(pooled: Pooled, metric: str) -> list[float] | None:
    if len(pooled.folds) < 2:
        return None
    if any(metric not in pooled.fold_metrics.get(index, {}) for index in pooled.folds):
        return None
    return [pooled.fold_metrics[index][metric] for index in sorted(pooled.folds)]


def _correspondence(candidate: Pooled, baseline: Pooled) -> str | None:
    """``None`` when two pooled runs may be compared; otherwise, why not.

    Three distinct mistakes, three distinct messages: a different split family, the same
    family read from a different store snapshot, and the same family and snapshot but a
    different set of folds. Matching on split family, store digest, and fold-index set
    additionally catches "fold 2 of split A compared against fold 2 of split B" -- a case
    the old whole-report fold fingerprint could not even be asked about, because it only
    existed once an in-process loop had finished every fold.
    """
    if candidate.split != baseline.split:
        return (
            f"different split families ({candidate.split!r} vs {baseline.split!r}); "
            "comparing across families measures the data, not the model"
        )
    if candidate.split_digest != baseline.split_digest:
        return (
            f"same split family {candidate.split!r} but different store snapshots "
            f"({candidate.split_digest[:12]} vs {baseline.split_digest[:12]}); a delta "
            "across store snapshots measures the data, not the model"
        )
    if set(candidate.folds) != set(baseline.folds):
        return (
            f"same split family and store snapshot but different folds "
            f"({sorted(candidate.folds)} vs {sorted(baseline.folds)}); a delta across "
            "different fold assignments measures the folds, not the models"
        )
    return None


def _classify(improvement: float, floor: float) -> Outcome:
    if improvement > floor:
        return Outcome.WIN
    if improvement < -floor:
        return Outcome.REGRESSION
    return Outcome.NEUTRAL


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
