"""Turning a ranking into a claim, or refusing to.

``dsio eval rank`` deliberately says a ranking is an ordering rather than a claim. This is
what makes the difference, by asking the three questions an ordering cannot answer:

1. **Is the winner better than luck?** The maximum of N noisy scores is above the truth by
   an amount that grows with N. Deflation subtracts it.
2. **Does the ranking transfer?** PBO asks whether picking the in-sample best would have
   picked well out of sample, over every symmetric split of the evidence.
3. **Was there enough evidence to tell?** A dependence-aware interval on the winner, and an
   effective sample size that does not pretend overlapping examples are independent.

The most valuable output is the negative one: *"a sweep of 200 configurations found nothing
that survives the correction."* That is a real answer, it is common, and a system that
cannot produce it will always hand back a winner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dsio.contracts import DsioModel
from dsio.eval.multiplicity import (
    BootstrapInterval,
    Deflation,
    MultiplicityError,
    PBOReport,
    block_bootstrap,
    deflate,
    pbo,
)


class Selection(DsioModel):
    """The verdict on a sweep: who won, and whether that means anything."""

    winner: str
    outcome: str
    deflation: Deflation
    overfitting: PBOReport | None = None
    interval: BootstrapInterval | None = None
    reason: str = ""
    runner_up: str | None = None
    runner_up_margin: float | None = None

    @property
    def selected(self) -> bool:
        return self.outcome == "selected"

    def summary_lines(self) -> list[str]:
        lines = [f"{self.outcome.upper()}: {self.winner} — {self.reason}"]
        lines += [f"  {line}" for line in self.deflation.summary_lines()]
        if self.overfitting is not None:
            lines += [f"  {line}" for line in self.overfitting.summary_lines()]
        if self.interval is not None:
            lines += [f"  {line}" for line in self.interval.summary_lines()]
        if self.runner_up is not None and self.runner_up_margin is not None:
            lines.append(
                f"  runner-up {self.runner_up} is {self.runner_up_margin:+.4f} behind"
            )
        return lines


def select(
    scores: Mapping[str, float],
    *,
    blocks: Mapping[str, Sequence[float]] | None = None,
    fold_sd: float | None = None,
    winner_values: Sequence[float] | None = None,
    n_trials: int | None = None,
    alpha: float = 0.05,
    pbo_threshold: float = 0.25,
    seed: int = 0,
) -> Selection:
    """Judge the best of ``scores`` against the search that produced it.

    ``blocks`` maps each configuration to its per-block scores — per fold, or per chunk of
    out-of-fold predictions. Supplying it enables both the PBO test and the estimate of how
    many trials were genuinely independent, and without it the verdict says so rather than
    quietly reporting a weaker check as if it were the full one.

    A winner is *selected* only when it clears the luck baseline, the search is unlikely
    enough under the null, and the ranking transfers. Failing any of the three is reported
    with which one failed.
    """
    if not scores:
        raise MultiplicityError("no scores to select from")

    matrix, names = _matrix(blocks, scores)
    result = deflate(dict(scores), trial_sd=fold_sd, blocks=matrix, n_trials=n_trials)

    overfitting: PBOReport | None = None
    if matrix is not None and matrix.shape[0] >= 2 and matrix.shape[1] >= 4:
        even = matrix[:, : matrix.shape[1] - (matrix.shape[1] % 2)]
        overfitting = pbo(even, seed=seed)

    interval: BootstrapInterval | None = None
    if winner_values is not None and len(winner_values) >= 4:
        interval = block_bootstrap(np.asarray(winner_values), seed=seed)
    elif matrix is not None and names is not None and result.winner in names:
        row = matrix[names.index(result.winner)]
        if row.size >= 4:
            interval = block_bootstrap(row, seed=seed)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    runner_up = ordered[1][0] if len(ordered) > 1 else None
    margin = ordered[1][1] - ordered[0][1] if len(ordered) > 1 else None

    outcome, reason = _judge(result, overfitting, alpha, pbo_threshold, blocks is not None)
    return Selection(
        winner=result.winner,
        outcome=outcome,
        deflation=result,
        overfitting=overfitting,
        interval=interval,
        reason=reason,
        runner_up=runner_up,
        runner_up_margin=margin,
    )


def _judge(
    result: Deflation,
    overfitting: PBOReport | None,
    alpha: float,
    pbo_threshold: float,
    had_blocks: bool,
) -> tuple[str, str]:
    """Three gates, reported by which one failed rather than as one opaque boolean."""
    if not result.survives:
        return (
            "not selected",
            f"the winner does not beat what luck alone would reach across "
            f"{result.n_trials} trial(s) ({result.best:.4f} vs {result.luck_baseline:.4f})",
        )
    if result.p_value > alpha:
        return (
            "not selected",
            f"a maximum this good would appear {result.p_value:.1%} of the time with no "
            f"real differences between configurations",
        )
    if overfitting is not None and overfitting.pbo > pbo_threshold:
        return (
            "not selected",
            f"the ranking does not transfer: PBO {overfitting.pbo:.1%} means picking the "
            "in-sample best is close to picking at random",
        )
    if not had_blocks:
        return (
            "provisional",
            "clears the luck baseline, but no per-block scores were supplied so the "
            "ranking's ability to transfer was never tested",
        )
    return (
        "selected",
        f"beats the luck baseline by {result.deflated:+.4f} (p={result.p_value:.3f}) and "
        f"the ranking transfers"
        + (f" (PBO {overfitting.pbo:.1%})" if overfitting is not None else ""),
    )


def _matrix(
    blocks: Mapping[str, Sequence[float]] | None, scores: Mapping[str, float]
) -> tuple[np.ndarray | None, list[str] | None]:
    """Arrange per-config block scores into ``[configs, blocks]``, in score-key order."""
    if blocks is None:
        return None, None
    names = sorted(scores)
    missing = [name for name in names if name not in blocks]
    if missing:
        raise MultiplicityError(
            f"no per-block scores for {', '.join(missing[:5])}; supply them for every "
            "configuration or omit blocks entirely — a partial matrix would silently test "
            "a different sweep than the one that was run"
        )
    widths = {len(blocks[name]) for name in names}
    if len(widths) != 1:
        raise MultiplicityError(
            f"configurations have different block counts ({sorted(widths)}); they must be "
            "scored on the same evidence to be ranked against each other"
        )
    return np.array([list(blocks[name]) for name in names], dtype=np.float64), names


def blocks_from_reports(reports: Mapping[str, Any], metric: str) -> dict[str, list[float]]:
    """Per-fold scores for each run, taken from its evaluation report.

    The natural block for a sweep: every run already records per-fold metrics in the shared
    artifact contract, and runs driven by the same committed splits are scored on identical
    folds — which is exactly the alignment PBO needs. Runs missing the metric are omitted
    rather than filled, because a fabricated block would be evidence nobody gathered.
    """
    out: dict[str, list[float]] = {}
    for name, report in reports.items():
        folds = getattr(report, "folds", ())
        values = [fold.metrics[metric] for fold in folds if metric in fold.metrics]
        if values and len(values) == len(folds):
            out[name] = values
    return out
