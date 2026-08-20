"""Is this result real? Baseline-relative verdicts, filtered by a noise floor.

**The rule.** An experiment is a win only if its improvement over its baseline exceeds the
noise floor. Otherwise it is neutral, however good the number looks. Chasing improvements
below the floor is how a week disappears into variance.

**The paired floor.** Estimating the floor from the candidate's own fold spread is the
best available when you cannot be sure two runs used the same folds. dsio commits its folds
and fingerprints the assignment, so when two runs *provably* held out the same examples the
far sharper comparison is available: the standard error of the per-fold *differences*.
Fold-to-fold variation the two models share — one fold simply being harder — cancels, and a
real 0.002 improvement becomes visible where the unpaired floor would have buried it under a
0.02 fold spread.

**The refusal.** Two runs with different fold assignments are not compared at all. A
refusal is worth far more than a confident, meaningless delta.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import Field

from dsio.contracts import DsioModel
from dsio.eval.contract import CVReport, EvalError

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
    spread. Prefer :func:`compare`, which takes the reports and can pair the folds.
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
    candidate: CVReport,
    baseline: CVReport,
    *,
    metric: str,
    higher_is_better: bool = True,
    k: float = DEFAULT_K,
    require_same_folds: bool = True,
) -> Comparison:
    """Judge two cross-validated runs against each other on one metric.

    Pairs the folds when both reports share a fold fingerprint, which is the normal case
    for two runs driven by the same committed split files. Falls back to the unpaired floor
    when one of them predates fingerprinting or the folds genuinely differ and the caller
    has said that is acceptable.
    """
    for report, label in ((candidate, "candidate"), (baseline, "baseline")):
        if metric not in report.metrics:
            return Comparison(
                metric=metric,
                outcome=Outcome.UNKNOWN,
                higher_is_better=higher_is_better,
                reason=(
                    f"{label} did not record {metric!r}; it has "
                    f"{', '.join(sorted(report.metrics)) or 'nothing'}"
                ),
            )

    comparable = _fold_structure_matches(candidate, baseline)
    if require_same_folds and not comparable:
        raise EvalError(
            f"refusing to compare runs on different fold assignments "
            f"({candidate.fold_fingerprint} vs {baseline.fold_fingerprint}). The fold "
            "assignment is the single source of truth; a delta across different folds "
            "measures the folds, not the models. Re-run against the same split files, or "
            "pass require_same_folds=False and accept the unpaired floor."
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
    candidate: CVReport,
    baseline: CVReport,
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


def _fold_series(report: CVReport, metric: str) -> list[float] | None:
    if len(report.folds) < 2:
        return None
    if any(metric not in fold.metrics for fold in report.folds):
        return None
    return [fold.metrics[metric] for fold in sorted(report.folds, key=lambda f: f.fold)]


def _fold_structure_matches(left: CVReport, right: CVReport) -> bool:
    if left.fold_fingerprint is None or right.fold_fingerprint is None:
        return False
    return left.fold_fingerprint == right.fold_fingerprint


def _classify(improvement: float, floor: float) -> Outcome:
    if improvement > floor:
        return Outcome.WIN
    if improvement < -floor:
        return Outcome.REGRESSION
    return Outcome.NEUTRAL


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
