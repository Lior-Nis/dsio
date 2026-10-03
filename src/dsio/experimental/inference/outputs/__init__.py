"""Prediction outputs and their parameter-matched validators."""

from dsio.experimental.inference.outputs.binary import BinaryOutput, BinaryValidator
from dsio.experimental.inference.outputs.contracts import PredictionViolation
from dsio.experimental.inference.outputs.multiclass import MulticlassOutput, MulticlassValidator
from dsio.experimental.inference.outputs.regression import RegressionOutput, RegressionValidator

__all__ = [
    "BinaryOutput",
    "BinaryValidator",
    "MulticlassOutput",
    "MulticlassValidator",
    "PredictionViolation",
    "RegressionOutput",
    "RegressionValidator",
]
