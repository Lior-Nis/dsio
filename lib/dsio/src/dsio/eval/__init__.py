"""Evaluation: the fold loop, the fixed artifact contract, and metrics.

This package is a leaf. It imports nothing from dsio except :mod:`dsio.contracts` and the
registry, which is what lets the fold loop drive an sklearn pipeline, a Lightning module or
a forecast model without any of them appearing in its type signatures.
"""

from dsio.eval.contract import (
    CVReport,
    EvalError,
    Fold,
    FoldMetrics,
    FoldPrediction,
    OutOfFold,
    fold_fingerprint,
    read_report,
    write_report,
)
from dsio.eval.ess import autocorrelation, effective_sample_size
from dsio.eval.loop import FitPredict, cross_validate
from dsio.eval.metrics import METRICS, MetricError, compute, metric
from dsio.eval.verdict import (
    Comparison,
    Outcome,
    compare,
    compare_all,
    minimum_detectable_rows,
    noise_floor,
    paired_noise_floor,
    sampling_noise,
    verdict,
)

__all__ = [
    "METRICS",
    "CVReport",
    "Comparison",
    "EvalError",
    "FitPredict",
    "Fold",
    "FoldMetrics",
    "FoldPrediction",
    "MetricError",
    "OutOfFold",
    "Outcome",
    "autocorrelation",
    "compare",
    "compare_all",
    "compute",
    "cross_validate",
    "effective_sample_size",
    "fold_fingerprint",
    "metric",
    "minimum_detectable_rows",
    "noise_floor",
    "paired_noise_floor",
    "read_report",
    "sampling_noise",
    "verdict",
    "write_report",
]
