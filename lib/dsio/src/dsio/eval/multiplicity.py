"""Selection under multiplicity: what the best of N results is actually worth.

A sweep manufactures its own bias. Pick the highest score out of N configurations and that
score is systematically too high, because it is the *maximum* of N noisy measurements rather
than the value of the best configuration. The inflation grows with how hard you searched,
which means a bigger sweep produces a more impressive and less trustworthy winner.

The size of it is not marginal. With a fold spread of 0.006 — an ordinary number — and 200
configurations that are all genuinely identical, the winner reports about +0.016 over the
truth. That is nearly three times the fold spread, and it is entirely selection.

Four tools, each answering a different question:

- :func:`expected_max` and :func:`deflate` — how good would the winner look if nothing
  worked? Subtract that before believing anything.
- :func:`pbo` — is the *procedure* that picked the winner better than a coin flip?
- :func:`block_bootstrap` — honest error bars when neighbouring examples are not
  independent evidence.
- :func:`effective_sample_size` — how many independent observations there really are.

The first three come from the quantitative-finance literature on backtest overfitting, where
this failure is expensive and therefore well studied. Nothing here is finance-specific: the
inputs are a score matrix and a fold spread, and the corrections apply to accuracy, AUC,
success rate or anything else. The one piece deliberately left behind is the Deflated Sharpe
Ratio's skew and kurtosis machinery, which is about return distributions rather than about
selection.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations

import numpy as np

from dsio.contracts import DsioModel

EULER = 0.5772156649015329


class MultiplicityError(ValueError):
    """Raised when a selection cannot be judged from what was supplied."""


# --- normal helpers, without a scipy dependency ---------------------------------------


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation), accurate to ~1e-9.

    Implemented rather than imported because ``dsio.eval`` is a leaf and scipy is not a
    dependency of the spine. ``test_normal_ppf_matches_scipy`` pins it against the real
    thing in the dev environment, so "we wrote our own" cannot drift into "ours is wrong".
    """
    if not 0.0 < p < 1.0:
        raise MultiplicityError(f"normal_ppf needs 0 < p < 1, got {p}")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1 - 0.02425

    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# --- deflation --------------------------------------------------------------------------


def expected_max(n_trials: int, mean: float = 0.0, sd: float = 1.0) -> float:
    """The score the winner reaches by luck alone, given ``n_trials`` equally good configs.

    Computed by numerical integration of ``E[max] = n * INT x phi(x) Phi(x)^(n-1) dx``
    rather than by the usual Gumbel approximation, which is off by ~0.05 sd at small
    ``n_trials`` — and small sweeps are exactly where a wrong baseline flips a verdict.
    Deterministic, so two reports of the same sweep agree exactly.
    """
    if n_trials < 1:
        raise MultiplicityError(f"n_trials must be at least 1, got {n_trials}")
    if sd < 0:
        raise MultiplicityError(f"sd must be non-negative, got {sd}")
    if n_trials == 1 or sd == 0:
        return mean
    return mean + sd * _standard_expected_max(n_trials)


def _standard_expected_max(n: int, grid: int = 40001, span: float = 12.0) -> float:
    """``E[max]`` of ``n`` standard normals, by trapezoid over a fixed grid."""
    x = np.linspace(-span, span, grid)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    return float(np.trapezoid(n * x * phi * cdf ** (n - 1), x))


def selection_p_value(best: float, n_trials: int, mean: float, sd: float) -> float:
    """Probability of seeing a maximum at least this good if every config were equal.

    Exact for independent normal trials: ``P(max < x) = Phi(z)**n``, so the tail is
    ``1 - Phi(z)**n``. A large sweep needs a far more extreme winner to reach the same
    p-value, which is the entire point.
    """
    if sd <= 0:
        return 0.0 if best > mean else 1.0
    tail = normal_cdf((best - mean) / sd) ** n_trials
    return float(max(0.0, min(1.0, 1.0 - tail)))


def effective_trials(scores: np.ndarray) -> float:
    """How many *independent* configurations a sweep really explored.

    Trials in a sweep are correlated: configs share structure, and they are scored on the
    same folds. Treating 200 correlated trials as 200 independent ones overstates the
    correction. Estimated from the mean pairwise correlation of the per-block score vectors,
    which is the same shape of adjustment as an effective sample size.

    Takes ``[n_configs, n_blocks]``.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise MultiplicityError(f"expected [configs, blocks], got shape {matrix.shape}")
    n_configs = matrix.shape[0]
    if n_configs < 2 or matrix.shape[1] < 2:
        return float(n_configs)

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix)
    correlation = np.nan_to_num(correlation, nan=0.0)
    off = correlation[~np.eye(n_configs, dtype=bool)]
    rho = float(np.clip(off.mean(), 0.0, 0.999))
    return float(n_configs / (1 + rho * (n_configs - 1)))


class Deflation(DsioModel):
    """What the winner's score is worth once the search is accounted for."""

    best: float
    winner: str = ""
    n_trials: int
    effective_trials: float
    trial_mean: float
    trial_sd: float
    luck_baseline: float = 0.0
    deflated: float = 0.0
    p_value: float = 1.0
    alpha: float = 0.05

    @property
    def survives(self) -> bool:
        """Whether the winner is *significantly* beyond luck, not merely above it.

        The gate is the p-value, not ``deflated > 0``. ``luck_baseline`` is the *mean* of
        the maximum's distribution, so a pure-noise sweep exceeds it about half the time by
        construction — using it as a boolean would make the correction a coin flip while
        looking like a test. It stays in the report because the *distance* is informative;
        it is just not the decision.
        """
        return self.p_value <= self.alpha

    def summary_lines(self) -> list[str]:
        return [
            f"best {self.best:.4f} over {self.n_trials} trial(s) "
            f"(~{self.effective_trials:.1f} independent)",
            f"  luck alone would reach {self.luck_baseline:.4f}; "
            f"deflated {self.deflated:+.4f}, p={self.p_value:.3f}",
        ]


def deflate(
    scores: dict[str, float],
    *,
    trial_sd: float | None = None,
    blocks: np.ndarray | None = None,
    n_trials: int | None = None,
    alpha: float = 0.05,
) -> Deflation:
    """Discount the winning score by what the search itself was worth.

    ``trial_sd`` is the spread a single configuration's score would show on its own — the
    fold spread — and defaults to the spread *across* configurations, which is the right
    fallback when nothing more specific is known. ``blocks`` supplies a
    ``[configs, blocks]`` matrix used to estimate how many of the trials were independent.
    """
    if not scores:
        raise MultiplicityError("no scores to select from")
    names = sorted(scores)
    values = np.array([scores[name] for name in names], dtype=np.float64)
    winner = names[int(np.argmax(values))]

    count = n_trials if n_trials is not None else len(names)
    independent = float(count)
    if blocks is not None:
        independent = min(independent, effective_trials(blocks))

    mean = float(values.mean())
    spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sd = float(trial_sd) if trial_sd is not None else spread

    baseline = expected_max(max(1, round(independent)), mean, sd)
    best = float(values.max())
    return Deflation(
        best=best,
        winner=winner,
        n_trials=count,
        effective_trials=independent,
        trial_mean=mean,
        trial_sd=sd,
        luck_baseline=baseline,
        deflated=best - baseline,
        p_value=selection_p_value(best, max(1, round(independent)), mean, sd),
        alpha=alpha,
    )


# --- probability of overfitting -------------------------------------------------------------


class PBOReport(DsioModel):
    """Whether the procedure that picked the winner beats a coin flip."""

    pbo: float
    n_configs: int
    n_blocks: int
    n_splits: int
    median_logit: float = 0.0
    median_oos_rank: float = 0.0
    degradation: float = 0.0

    @property
    def reliable(self) -> bool:
        """Whether the matrix is big enough for the estimate to mean much.

        Measured, not guessed: under the null a 12x12 matrix returns PBO anywhere from 0.08
        to 0.92 depending on the draw, because every split reuses the same fixed scores.
        Below roughly 10 configs and 10 blocks a single PBO figure is close to unreadable,
        and reporting it without that caveat invites exactly the overconfidence the
        statistic exists to prevent.
        """
        return self.n_configs >= 10 and self.n_blocks >= 10

    @property
    def verdict(self) -> str:
        if self.pbo >= 0.5:
            return "no better than chance"
        if self.pbo >= 0.25:
            return "weak"
        return "selection transfers"

    def summary_lines(self) -> list[str]:
        lines = [
            f"PBO {self.pbo:.2%} over {self.n_splits} symmetric splits of "
            f"{self.n_blocks} blocks — {self.verdict}",
            f"  the in-sample winner lands at rank {self.median_oos_rank:.2f} "
            f"out of sample (0.5 = coin flip)",
        ]
        if not self.reliable:
            lines.append(
                f"  {self.n_configs} configs x {self.n_blocks} blocks is small; a single "
                "PBO estimate at this size is itself very noisy"
            )
        return lines


def pbo(
    scores: np.ndarray,
    *,
    n_blocks: int | None = None,
    max_splits: int = 20_000,
    seed: int = 0,
) -> PBOReport:
    """Probability of Backtest Overfitting, via combinatorially symmetric cross-validation.

    Takes a ``[configs, blocks]`` performance matrix. For every way of splitting the blocks
    into two equal halves, find the configuration that ranks best on one half and see where
    it lands on the other. PBO is the fraction of splits where the in-sample winner falls
    into the *bottom* half out of sample.

    Near 0.5 means selecting the best config is a coin flip: whatever the sweep found was
    noise, and its ranking will not survive contact with new data. Near 0 means the ranking
    genuinely transfers.

    Symmetric — every split is also used with its halves swapped — because an asymmetric
    version measures which half happened to be easier as well as the thing it is after.

    The split count is ``C(blocks, blocks/2)``, which is 12,870 at 16 blocks and explodes
    after that; beyond ``max_splits`` a deterministic random sample is used instead, and the
    count is reported so nobody reads a sampled estimate as an exhaustive one.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise MultiplicityError(f"expected [configs, blocks], got shape {matrix.shape}")
    n_configs, available = matrix.shape
    if n_configs < 2:
        raise MultiplicityError("PBO needs at least two configurations to rank")

    blocks = n_blocks or available
    if blocks % 2:
        raise MultiplicityError(f"need an even number of blocks to halve them, got {blocks}")
    if blocks != available:
        raise MultiplicityError(f"matrix has {available} blocks, but {blocks} was requested")
    if blocks < 4:
        raise MultiplicityError(
            f"{blocks} blocks gives too few symmetric splits to estimate PBO; use at least 4"
        )

    half = blocks // 2
    every = list(combinations(range(blocks), half))
    rng = np.random.default_rng(seed)
    if len(every) > max_splits:
        chosen = rng.choice(len(every), size=max_splits, replace=False)
        every = [every[int(i)] for i in sorted(chosen)]

    logits: list[float] = []
    ranks: list[float] = []
    degradations: list[float] = []

    for train_blocks in every:
        mask = np.zeros(blocks, dtype=bool)
        mask[list(train_blocks)] = True
        in_sample = matrix[:, mask].mean(axis=1)
        out_sample = matrix[:, ~mask].mean(axis=1)

        best = int(np.argmax(in_sample))
        # Relative rank of that config out of sample, in (0, 1). Ties share their average
        # rank, so a matrix of identical configs sits exactly at 0.5 rather than at an
        # arbitrary end of the ordering.
        order = _average_rank(out_sample)
        omega = order[best] / (n_configs + 1)
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits.append(math.log(omega / (1 - omega)))
        ranks.append(omega)
        degradations.append(float(out_sample[best] - np.median(out_sample)))

    array = np.array(logits)
    return PBOReport(
        pbo=float(np.mean(array <= 0.0)),
        n_configs=n_configs,
        n_blocks=blocks,
        n_splits=len(every),
        median_logit=float(np.median(array)),
        median_oos_rank=float(np.median(ranks)),
        degradation=float(np.median(degradations)),
    )


def _average_rank(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    per_group = first + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(per_group, counts)
    return ranks


# --- dependence-aware resampling ------------------------------------------------------------


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag."""
    series = np.asarray(values, dtype=np.float64)
    n = series.size
    if n < 3:
        raise MultiplicityError("need at least three observations for autocorrelation")
    lag_limit = max_lag or min(n // 2, 200)
    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0:
        return np.zeros(lag_limit)
    return np.array(
        [float(np.dot(centred[:-k], centred[k:]) / denominator) for k in range(1, lag_limit + 1)]
    )


def effective_sample_size(values: np.ndarray, max_lag: int | None = None) -> float:
    """How many independent observations a correlated series is worth.

    Overlapping windows are the obvious case: at length 500 and stride 200, adjacent windows
    share 60% of their rows, so ten thousand of them are nothing like ten thousand
    independent measurements. Every confidence interval computed as if they were is too
    narrow, and the error compounds into every comparison built on it.

    Uses the initial positive sequence — summing autocorrelations until the first negative
    one — which is the standard truncation and avoids accumulating noise from long lags.
    """
    rho = autocorrelation(values, max_lag)
    total = 0.0
    for value in rho:
        if value <= 0:
            break
        total += value
    n = float(np.asarray(values).size)
    return float(n / (1.0 + 2.0 * total))


class BootstrapInterval(DsioModel):
    """A confidence interval that respects the dependence in the data."""

    estimate: float
    low: float
    high: float
    confidence: float
    block_size: int
    n_resamples: int
    effective_n: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def summary_lines(self) -> list[str]:
        return [
            f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}] "
            f"at {self.confidence:.0%}, block size {self.block_size}",
            f"  {self.effective_n:.0f} effective observations",
        ]


def block_bootstrap(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] | None = None,
    *,
    block_size: int | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Circular block bootstrap: resample contiguous runs, not individual observations.

    The ordinary bootstrap resamples rows independently, which assumes each row is
    independent evidence. When neighbouring observations are correlated that assumption
    produces intervals that are far too narrow — confidently wrong rather than merely
    uncertain. Resampling contiguous blocks preserves the local dependence.

    Circular rather than plain: wrapping the series means every observation appears in the
    same number of candidate blocks, where a non-circular version under-samples both ends.

    ``block_size`` defaults to ``n^(1/3)`` inflated by the measured dependence, so a series
    with no autocorrelation gets small blocks and a strongly dependent one gets long ones.
    """
    series = np.asarray(values, dtype=np.float64)
    n = series.size
    if n < 4:
        raise MultiplicityError(f"need at least four observations to bootstrap, got {n}")
    if not 0 < confidence < 1:
        raise MultiplicityError(f"confidence must be in (0, 1), got {confidence}")

    compute = statistic or (lambda sample: float(np.mean(sample)))
    ess = effective_sample_size(series)
    size = block_size or max(1, min(n // 2, round((n ** (1 / 3)) * max(1.0, n / ess))))

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / size)
    doubled = np.concatenate([series, series])

    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([doubled[s : s + size] for s in starts])[:n]
        estimates[i] = compute(sample)

    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=float(compute(series)),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1 - alpha)),
        confidence=confidence,
        block_size=size,
        n_resamples=n_resamples,
        effective_n=ess,
    )
