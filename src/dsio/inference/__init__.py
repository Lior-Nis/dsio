"""Portable, schema-validated inference without serving or deployment policy."""

from dsio.inference.export import ExportError, ExportForm, log_predictor
from dsio.inference.lineage import require_checkpoint_lineage
from dsio.inference.loading import InferenceError, predict
from dsio.inference.predictor import (
    Predictor,
    PredictorError,
    TensorOutput,
    build_predictor,
    validate_tensor_prediction,
)

__all__ = [
    "ExportError",
    "ExportForm",
    "InferenceError",
    "Predictor",
    "PredictorError",
    "TensorOutput",
    "build_predictor",
    "log_predictor",
    "predict",
    "require_checkpoint_lineage",
    "validate_tensor_prediction",
]
