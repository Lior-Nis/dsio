"""Effective sample size for autocorrelated evidence.

Overlapping windows are not independent examples. A metric computed over 10,000
windows drawn with stride 100 from a 500-step window has nothing like 10,000
independent observations, and any interval that assumes it does is too narrow.
This is kept from the deleted multiplicity layer because it is true at any number
of trials, where deflation and PBO need hundreds to say anything.
"""

from __future__ import annotations

import numpy as np


class EssError(ValueError):
    """Raised when effective sample size cannot be computed from what was supplied."""


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Sample autocorrelation at lags 0..max_lag. Lag 0 is 1 by definition."""
    series = np.asarray(values, dtype=np.float64)
    n = series.size
    if n < 3:
        raise EssError("need at least three observations for autocorrelation")
    lag_limit = max_lag or min(n // 2, 200)
    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0:
        return np.concatenate(([1.0], np.zeros(lag_limit)))
    lags = np.array(
        [float(np.dot(centred[:-k], centred[k:]) / denominator) for k in range(1, lag_limit + 1)]
    )
    return np.concatenate(([1.0], lags))


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
    for value in rho[1:]:
        if value <= 0:
            break
        total += value
    n = float(np.asarray(values).size)
    return float(n / (1.0 + 2.0 * total))
