import numpy as np
import pytest

from dsio.eval.ess import EssError, autocorrelation, effective_sample_size


def test_independent_series_has_ess_close_to_n():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert effective_sample_size(x) == pytest.approx(4000, rel=0.25)


def test_correlated_series_has_ess_below_n():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=4000)
    x = np.convolve(noise, np.ones(20) / 20, mode="same")
    assert effective_sample_size(x) < 1000


def test_autocorrelation_at_lag_zero_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert autocorrelation(x, max_lag=5)[0] == pytest.approx(1.0)


def test_ess_never_exceeds_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    assert effective_sample_size(x) <= 200


def test_autocorrelation_raises_below_three_observations():
    with pytest.raises(EssError, match="at least three observations"):
        autocorrelation(np.array([1.0, 2.0]))


def test_constant_series_has_zero_autocorrelation_beyond_lag_zero():
    # A constant series has zero variance, so the denominator in the autocorrelation
    # sum is zero; this must return a valid (zero) result rather than divide by zero.
    x = np.full(10, 5.0)
    rho = autocorrelation(x, max_lag=4)
    assert rho[0] == pytest.approx(1.0)
    assert np.all(rho[1:] == 0.0)
    assert effective_sample_size(x, max_lag=4) == pytest.approx(10.0)
