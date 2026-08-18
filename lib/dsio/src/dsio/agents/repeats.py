"""The second noise floor: the model's own sampling variance.

Everywhere else in dsio, uncertainty comes from which examples landed in which fold. An
agent evaluation has a second, independent source, and it is usually the larger one: run the
*same* configuration on the *same* tasks twice and get a different number, because the model
sampled differently.

Ignoring it is the characteristic error of agent benchmarking. A prompt change that moves
success from 62% to 65% looks like an improvement, and on a hundred tasks at temperature 1.0
the run-to-run spread is frequently wider than that. Reporting the delta without the spread
is the same mistake as reporting a fold delta without the fold spread — which dsio already
refuses to do.

The estimator is deliberately the simplest one that answers the question people actually
ask. Take repeat *j* of every task; that is one complete evaluation pass. Do it for each of
the *k* repeats and take the standard deviation across those passes. The result is literally
"if I ran this whole evaluation again, how much would the headline move?"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from dsio.contracts import DsioModel


class RepeatError(ValueError):
    """Raised when repeats are missing or ragged."""


class RepeatReport(DsioModel):
    """What repeating an evaluation revealed about its stability."""

    n_examples: int
    n_repeats: int
    mean: float
    run_std: float
    run_values: tuple[float, ...] = ()
    unstable_examples: tuple[str, ...] = ()
    agreement: float = 1.0

    @property
    def usable(self) -> bool:
        """Whether the spread was measured at all. One repeat cannot see it."""
        return self.n_repeats >= 2

    def summary_lines(self) -> list[str]:
        if not self.usable:
            return [
                f"{self.mean:.4f} over {self.n_examples} examples, 1 repeat "
                "(model sampling noise not measured)"
            ]
        return [
            f"{self.mean:.4f} +/- {self.run_std:.4f} across {self.n_repeats} full passes "
            f"of {self.n_examples} examples",
            f"  {len(self.unstable_examples)} example(s) did not answer consistently "
            f"({self.agreement:.1%} agreement)",
        ]


def as_matrix(
    outcomes: Mapping[str, Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Arrange per-example repeats into an ``[examples, repeats]`` matrix.

    Ragged input is rejected rather than padded. A task that failed to produce one of its
    repeats is missing evidence, and filling the hole with a zero would report it as a
    failure the model never had.
    """
    if isinstance(outcomes, np.ndarray):
        matrix = np.asarray(outcomes, dtype=np.float64)
        if matrix.ndim != 2:
            raise RepeatError(f"expected [examples, repeats], got shape {matrix.shape}")
        return matrix, [str(i) for i in range(matrix.shape[0])]

    if not outcomes:
        raise RepeatError("no outcomes to summarise")
    names = sorted(outcomes)
    widths = {len(outcomes[name]) for name in names}
    if len(widths) != 1:
        ragged = [name for name in names if len(outcomes[name]) != max(widths)]
        raise RepeatError(
            f"repeat counts differ across examples: {', '.join(ragged[:5])} have fewer than "
            f"{max(widths)}. A missing repeat is missing evidence, not a failure."
        )
    return np.array([list(outcomes[name]) for name in names], dtype=np.float64), names


def repeat_report(outcomes: Mapping[str, Sequence[float]] | np.ndarray) -> RepeatReport:
    """Summarise an ``[examples, repeats]`` outcome matrix.

    ``run_std`` is the spread across complete passes and is the number to compare a claimed
    improvement against. ``agreement`` is the fraction of examples that answered the same
    way every time, which localises the instability: a low run_std with low agreement means
    individual tasks are flipping but cancelling out, and that is a different problem from
    a uniformly noisy system.
    """
    matrix, names = as_matrix(outcomes)
    n_examples, n_repeats = matrix.shape
    if n_examples == 0:
        raise RepeatError("no examples to summarise")

    per_pass = matrix.mean(axis=0)
    consistent = np.all(matrix == matrix[:, :1], axis=1)
    unstable = [name for name, ok in zip(names, consistent, strict=True) if not ok]

    return RepeatReport(
        n_examples=int(n_examples),
        n_repeats=int(n_repeats),
        mean=float(matrix.mean()),
        run_std=float(np.std(per_pass, ddof=1)) if n_repeats >= 2 else 0.0,
        run_values=tuple(float(value) for value in per_pass),
        unstable_examples=tuple(unstable),
        agreement=float(consistent.mean()),
    )


def combined_floor(fold_std: float, run_std: float) -> float:
    """Fold spread and sampling spread added in quadrature.

    They are independent sources, so the floor a claimed improvement has to clear is the
    root of the sum of squares rather than either one alone. Using only the fold spread —
    which is what happens when an agent evaluation is dropped into ordinary
    cross-validation machinery — understates the bar, usually by a lot.
    """
    return float(np.hypot(fold_std, run_std))


def stability_verdict(
    candidate: RepeatReport,
    baseline: RepeatReport,
    *,
    k: float = 1.0,
) -> dict[str, float | str | bool]:
    """Judge two agent evaluations against their combined sampling spread.

    Refuses to call an improvement real when it sits inside the run-to-run noise of either
    arm, which is the specific check missing from most agent benchmark comparisons.
    """
    improvement = candidate.mean - baseline.mean
    floor = k * float(np.hypot(candidate.run_std, baseline.run_std))
    if not (candidate.usable and baseline.usable):
        return {
            "improvement": improvement,
            "noise_floor": 0.0,
            "outcome": "unknown",
            "measured": False,
            "reason": (
                "at least one arm has a single repeat, so model sampling noise was never "
                "measured; any delta here is unqualified"
            ),
        }
    return {
        "improvement": improvement,
        "noise_floor": floor,
        "outcome": (
            "win" if improvement > floor else "regression" if improvement < -floor else "neutral"
        ),
        "measured": True,
        "reason": f"judged against {k} sd of run-to-run spread",
    }
