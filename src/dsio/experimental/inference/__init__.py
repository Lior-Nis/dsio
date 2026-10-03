"""Experimental inference-side components: prediction outputs and their validators.

Each block's maturity and evidence is listed in docs/component-warehouse/catalog.md.
"""

from dsio.experimental.inference.outputs import (
    BinaryOutput,
    BinaryValidator,
    MulticlassOutput,
    MulticlassValidator,
    PredictionViolation,
    RegressionOutput,
    RegressionValidator,
)

__all__ = [
    "BinaryOutput",
    "BinaryValidator",
    "MulticlassOutput",
    "MulticlassValidator",
    "PredictionViolation",
    "RegressionOutput",
    "RegressionValidator",
]
