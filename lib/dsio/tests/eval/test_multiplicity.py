"""Selection under multiplicity.

The tests that matter are the honesty ones: given a sweep where nothing is real, the
machinery must say so — and given a sweep where something is, it must not cry wolf. Both
directions are checked against simulated ground truth, because a correction that is merely
conservative is as useless as one that is merely permissive.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsio.eval import (
    MultiplicityError,
    autocorrelation,
    block_bootstrap,
    deflate,
    effective_sample_size,
    effective_trials,
    expected_max,
    pbo,
    select,
    selection_p_value,
)
from dsio.eval.multiplicity import normal_cdf, normal_ppf


def noisy_sweep(
    n_configs: int,
    n_blocks: int,
    *,
    truth: np.ndarray | float = 0.0,
    sd: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """A [configs, blocks] matrix of measurements around a per-config true value."""
    rng = np.random.default_rng(seed)
    centre = np.full(n_configs, truth) if np.isscalar(truth) else np.asarray(truth)
    return centre[:, None] + rng.normal(0, sd, size=(n_configs, n_blocks))


# --- the normal helpers -----------------------------------------------------------------


def test_normal_ppf_matches_scipy() -> None:
    """dsio.eval is a leaf and scipy is not a dependency, so this is hand-rolled — which is
    only defensible if it agrees with the real thing."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for p in (1e-6, 0.001, 0.02, 0.1, 0.5, 0.9, 0.975, 0.999, 1 - 1e-6):
        assert normal_ppf(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=1e-6)


def test_normal_cdf_and_ppf_are_inverses() -> None:
    for z in (-3.0, -0.5, 0.0, 1.2, 2.8):
        assert normal_ppf(normal_cdf(z)) == pytest.approx(z, abs=1e-6)


def test_normal_ppf_rejects_a_degenerate_probability() -> None:
    with pytest.raises(MultiplicityError, match="0 < p < 1"):
        normal_ppf(0.0)


# --- deflation ---------------------------------------------------------------------------


def test_expected_max_matches_simulation() -> None:
    """Integrated rather than approximated, so a report is deterministic; simulation is how
    we know it is the right integral.

    The usual Gumbel approximation is off by ~0.05 sd at n=2, and small sweeps are exactly
    where a wrong baseline flips a verdict — hence the tolerance here is 0.01, which the
    approximation would not meet.
    """
    rng = np.random.default_rng(0)
    for n in (2, 10, 50, 200, 1000):
        simulated = rng.normal(0, 1, size=(60000, n)).max(axis=1).mean()
        assert expected_max(n) == pytest.approx(simulated, abs=0.01)


def test_expected_max_grows_with_the_size_of_the_search() -> None:
    """The whole premise: searching harder inflates the winner more."""
    values = [expected_max(n, 0.0, 1.0) for n in (1, 5, 50, 500)]
    assert values == sorted(values)
    assert values[0] == 0.0


def test_one_trial_has_nothing_to_deflate() -> None:
    assert expected_max(1, 0.9, 0.01) == 0.9


def test_a_pure_noise_sweep_does_not_survive_deflation() -> None:
    """200 configurations that are genuinely identical, over many independent sweeps.

    The best of them always looks impressive and never means anything. A correct
    correction rejects it at close to the nominal rate — measured across sweeps, because a
    single sweep proves nothing about a statistical gate.
    """
    survived = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        scores = {f"c{i}": 0.9 + rng.normal(0, 0.01) for i in range(200)}
        survived += deflate(scores).survives
    assert survived <= 6, f"{survived}/60 pure-noise sweeps were accepted"


def test_the_luck_baseline_alone_would_be_a_coin_flip() -> None:
    """Why the gate is the p-value and not ``deflated > 0``.

    ``luck_baseline`` is the *mean* of the maximum's distribution, so a pure-noise sweep
    exceeds it about half the time by construction. Using it as a boolean would make the
    correction a coin flip while looking like a test — which is worse than no correction,
    because it carries authority.
    """
    above = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        scores = {f"c{i}": 0.9 + rng.normal(0, 0.01) for i in range(200)}
        above += deflate(scores).deflated > 0
    assert 15 <= above <= 45, f"expected roughly half, got {above}/60"


def test_a_genuinely_better_config_does_survive() -> None:
    """The other direction. A correction that rejects everything is not a correction."""
    rng = np.random.default_rng(2)
    scores = {f"c{i}": 0.9 + rng.normal(0, 0.01) for i in range(50)}
    scores["winner"] = 0.99
    result = deflate(scores)
    assert result.winner == "winner"
    assert result.survives is True
    assert result.p_value < 0.01


def test_the_same_winner_survives_a_small_search_and_not_a_huge_one() -> None:
    """Identical evidence, different amounts of searching — the correction must move."""
    scores = {f"c{i}": 0.90 + 0.0005 * i for i in range(20)}
    small = deflate(scores, n_trials=20)
    huge = deflate(scores, n_trials=100_000)
    assert small.luck_baseline < huge.luck_baseline
    assert huge.deflated < small.deflated


def test_the_p_value_loosens_as_the_search_grows() -> None:
    """The same winning score is less surprising the harder you searched for it."""
    values = [selection_p_value(best=2.0, n_trials=n, mean=0.0, sd=1.0) for n in (1, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0228, abs=1e-3)
    assert values[-1] > 0.9


def test_correlated_trials_count_for_less_than_independent_ones() -> None:
    """Configs in a sweep share structure and share folds. Treating 200 correlated trials
    as 200 independent ones overstates the correction."""
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 1, size=(1, 12))
    correlated = np.repeat(shared, 20, axis=0) + rng.normal(0, 0.05, size=(20, 12))
    independent = rng.normal(0, 1, size=(20, 12))

    assert effective_trials(correlated) < 3.0
    assert effective_trials(independent) > 12.0


def test_deflation_reports_what_it_assumed() -> None:
    scores = {f"c{i}": float(i) for i in range(10)}
    result = deflate(scores, trial_sd=2.0)
    assert result.n_trials == 10 and result.trial_sd == 2.0
    assert "luck alone" in " ".join(result.summary_lines())


def test_deflating_nothing_is_an_error() -> None:
    with pytest.raises(MultiplicityError, match="no scores"):
        deflate({})


# --- probability of backtest overfitting --------------------------------------------------


def test_pbo_is_a_coin_flip_when_every_config_is_identical() -> None:
    """The headline check, averaged over independent sweeps.

    If all configs are equally good, picking the in-sample best is picking at random and
    PBO must centre on 0.5. It is averaged because every split of one matrix reuses the
    same fixed scores, so a single estimate is high-variance — see
    ``test_a_single_pbo_estimate_on_a_small_matrix_is_noisy``.
    """
    values = [pbo(noisy_sweep(12, 12, sd=1.0, seed=seed), seed=seed).pbo for seed in range(40)]
    assert float(np.mean(values)) == pytest.approx(0.5, abs=0.08)


def test_a_single_pbo_estimate_on_a_small_matrix_is_noisy() -> None:
    """A measured limitation, pinned so it is not rediscovered as a bug.

    Under the null a 12x12 matrix returns PBO anywhere from roughly 0.08 to 0.92 depending
    on the draw. The report says so via `reliable`, because a PBO figure quoted without
    that caveat invites exactly the overconfidence the statistic exists to prevent.
    """
    values = [pbo(noisy_sweep(12, 12, sd=1.0, seed=seed), seed=seed).pbo for seed in range(40)]
    assert max(values) - min(values) > 0.5
    assert pbo(noisy_sweep(12, 12, seed=0)).reliable is True
    assert pbo(noisy_sweep(6, 8, seed=0)).reliable is False


def test_pbo_is_near_zero_when_one_config_is_genuinely_best() -> None:
    """The other direction, so the statistic is not merely always 0.5."""
    truth = np.zeros(12)
    truth[0] = 5.0
    report = pbo(noisy_sweep(12, 12, truth=truth, sd=1.0, seed=5))
    assert report.pbo < 0.05
    assert report.verdict == "selection transfers"
    assert report.degradation > 0


def test_pbo_uses_every_symmetric_split() -> None:
    """C(8,4) = 70. Symmetric means each split is also used with its halves swapped, so a
    result cannot measure which half happened to be easier."""
    report = pbo(noisy_sweep(6, 8, seed=6))
    assert report.n_splits == 70
    assert report.n_blocks == 8


def test_pbo_samples_deterministically_when_the_split_count_explodes() -> None:
    """C(20,10) is 184,756. A sampled estimate is fine; silently reporting it as exhaustive
    is not, so the count comes back in the report."""
    matrix = noisy_sweep(5, 20, seed=7)
    report = pbo(matrix, max_splits=500)
    assert report.n_splits == 500
    assert pbo(matrix, max_splits=500).pbo == report.pbo


def test_pbo_rejects_shapes_it_cannot_halve() -> None:
    with pytest.raises(MultiplicityError, match="even number of blocks"):
        pbo(noisy_sweep(4, 7))
    with pytest.raises(MultiplicityError, match="at least 4"):
        pbo(noisy_sweep(4, 2))
    with pytest.raises(MultiplicityError, match="at least two configurations"):
        pbo(noisy_sweep(1, 8))


# --- dependence ---------------------------------------------------------------------------


def test_effective_sample_size_discounts_a_correlated_series() -> None:
    """Overlapping windows are the obvious case: adjacent examples share most of their
    rows, so ten thousand of them are nothing like ten thousand measurements."""
    rng = np.random.default_rng(8)
    independent = rng.normal(size=2000)
    smoothed = np.convolve(rng.normal(size=2200), np.ones(50) / 50, mode="valid")[:2000]

    assert effective_sample_size(independent) == pytest.approx(2000, rel=0.25)
    assert effective_sample_size(smoothed) < 300


def test_autocorrelation_of_white_noise_is_near_zero() -> None:
    rng = np.random.default_rng(9)
    assert abs(autocorrelation(rng.normal(size=5000), max_lag=5)).max() < 0.1


def test_block_bootstrap_is_wider_than_it_would_be_if_rows_were_independent() -> None:
    """The failure this prevents: an interval that is confidently wrong rather than
    honestly uncertain."""
    rng = np.random.default_rng(10)
    correlated = np.convolve(rng.normal(size=1100), np.ones(40) / 40, mode="valid")[:1000]

    blocked = block_bootstrap(correlated, block_size=80, seed=0)
    naive = block_bootstrap(correlated, block_size=1, seed=0)
    assert blocked.width > naive.width * 2


def test_block_bootstrap_covers_the_truth_for_independent_data() -> None:
    """Calibration: a 95% interval on well-behaved data should contain the true mean."""
    rng = np.random.default_rng(11)
    covered = 0
    for seed in range(40):
        sample = rng.normal(loc=1.0, scale=1.0, size=400)
        interval = block_bootstrap(sample, n_resamples=400, seed=seed)
        covered += interval.low <= 1.0 <= interval.high
    assert covered >= 34


def test_block_size_defaults_to_the_measured_dependence() -> None:
    rng = np.random.default_rng(12)
    independent = rng.normal(size=1000)
    correlated = np.convolve(rng.normal(size=1100), np.ones(50) / 50, mode="valid")[:1000]
    assert block_bootstrap(correlated, n_resamples=200).block_size > block_bootstrap(
        independent, n_resamples=200
    ).block_size


def test_bootstrap_accepts_any_statistic() -> None:
    rng = np.random.default_rng(13)
    values = rng.normal(size=500)
    interval = block_bootstrap(values, statistic=lambda s: float(np.median(s)), n_resamples=300)
    assert interval.low < interval.estimate < interval.high


def test_bootstrap_rejects_a_series_too_short_to_resample() -> None:
    with pytest.raises(MultiplicityError, match="at least four"):
        block_bootstrap(np.array([1.0, 2.0]))


# --- the composite verdict ------------------------------------------------------------------


def test_a_sweep_of_pure_noise_is_not_selected() -> None:
    """The most valuable output in the module: 'your sweep found nothing'."""
    matrix = noisy_sweep(30, 12, truth=0.0, sd=1.0, seed=14)
    scores = {f"c{i}": float(row.mean()) for i, row in enumerate(matrix)}
    blocks = {f"c{i}": row.tolist() for i, row in enumerate(matrix)}

    result = select(scores, blocks=blocks)
    assert result.selected is False
    assert result.outcome == "not selected"


def test_a_real_winner_is_selected() -> None:
    truth = np.zeros(20)
    truth[7] = 4.0
    matrix = noisy_sweep(20, 12, truth=truth, sd=1.0, seed=15)
    scores = {f"c{i}": float(row.mean()) for i, row in enumerate(matrix)}
    blocks = {f"c{i}": row.tolist() for i, row in enumerate(matrix)}

    result = select(scores, blocks=blocks)
    assert result.winner == "c7"
    assert result.selected is True
    assert result.overfitting is not None and result.overfitting.pbo < 0.25


def test_without_per_block_scores_the_verdict_is_provisional_not_confident() -> None:
    """A weaker check reported as if it were the full one is the failure mode here."""
    scores = {f"c{i}": 0.5 for i in range(10)}
    scores["winner"] = 5.0
    result = select(scores)
    assert result.outcome == "provisional"
    assert "never tested" in result.reason


def test_the_verdict_says_which_gate_failed() -> None:
    matrix = noisy_sweep(30, 12, truth=0.0, sd=1.0, seed=16)
    scores = {f"c{i}": float(row.mean()) for i, row in enumerate(matrix)}
    blocks = {f"c{i}": row.tolist() for i, row in enumerate(matrix)}
    result = select(scores, blocks=blocks)
    assert any(
        phrase in result.reason
        for phrase in ("luck alone", "would appear", "does not transfer")
    )


def test_a_partial_block_matrix_is_refused() -> None:
    """Testing a different sweep than the one that ran, silently, is the worst outcome."""
    with pytest.raises(MultiplicityError, match="no per-block scores"):
        select({"a": 1.0, "b": 2.0}, blocks={"a": [1.0, 1.0, 1.0, 1.0]})


def test_ragged_blocks_are_refused() -> None:
    with pytest.raises(MultiplicityError, match="different block counts"):
        select(
            {"a": 1.0, "b": 2.0},
            blocks={"a": [1.0] * 4, "b": [1.0] * 6},
        )


def test_the_report_names_the_runner_up_and_its_margin() -> None:
    """The gap at the top is what people read as meaningful; showing it next to the noise
    floor is what makes it readable."""
    matrix = noisy_sweep(10, 8, seed=17)
    scores = {f"c{i}": float(row.mean()) for i, row in enumerate(matrix)}
    result = select(scores, blocks={f"c{i}": row.tolist() for i, row in enumerate(matrix)})
    assert result.runner_up is not None
    assert result.runner_up_margin is not None and result.runner_up_margin <= 0
    assert result.summary_lines()
