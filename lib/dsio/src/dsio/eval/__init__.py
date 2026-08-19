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
from dsio.eval.loop import FitPredict, cross_validate
from dsio.eval.metrics import METRICS, MetricError, compute, metric
from dsio.eval.multiplicity import (
    BootstrapInterval,
    Deflation,
    MultiplicityError,
    PBOReport,
    autocorrelation,
    block_bootstrap,
    deflate,
    effective_sample_size,
    effective_trials,
    expected_max,
    pbo,
    selection_p_value,
)
from dsio.eval.select import Selection, blocks_from_reports, select
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
    "BootstrapInterval",
    "CVReport",
    "Comparison",
    "Deflation",
    "EvalError",
    "FitPredict",
    "Fold",
    "FoldMetrics",
    "FoldPrediction",
    "MetricError",
    "MultiplicityError",
    "OutOfFold",
    "Outcome",
    "PBOReport",
    "Selection",
    "autocorrelation",
    "block_bootstrap",
    "blocks_from_reports",
    "compare",
    "compare_all",
    "compute",
    "cross_validate",
    "deflate",
    "effective_sample_size",
    "effective_trials",
    "expected_max",
    "fold_fingerprint",
    "metric",
    "minimum_detectable_rows",
    "noise_floor",
    "paired_noise_floor",
    "pbo",
    "read_report",
    "sampling_noise",
    "select",
    "selection_p_value",
    "verdict",
    "write_report",
]
