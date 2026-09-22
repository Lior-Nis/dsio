"""Portable, schema-validated inference without serving or deployment policy."""

from dsio.inference.predictor import (
    Predictor,
    PredictorError,
    TensorOutput,
    build_predictor,
    validate_tensor_prediction,
)

__all__ = [
    "Predictor",
    "PredictorError",
    "TensorOutput",
    "build_predictor",
    "validate_tensor_prediction",
]
