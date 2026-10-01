"""Tensor output naming and finiteness validation (legacy experimental).

Legacy experimental under the pre-1.0 clause in docs/component-admission.md: used only by
the synthetic reference fixtures, no compatibility promise. The warehouse regression output
(Component Warehouse v1, Story 7.5) supersedes it, and it is deleted once unused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.inference.predictor import PredictorError


class TensorOutput(nn.Module):
    """Name one native tensor model output without changing it."""

    def __init__(self, field: str = "prediction") -> None:
        super().__init__()
        if not isinstance(field, str) or not field or field == "sample_id":
            raise PredictorError("tensor output field must be non-empty and not 'sample_id'")
        self.field = field

    def forward(self, value: Any) -> Mapping[str, Tensor]:
        if not isinstance(value, Tensor):
            raise PredictorError(
                f"tensor output normalizer expected a tensor, got {type(value).__name__}"
            )
        return {self.field: value}


def validate_tensor_prediction(output: Mapping[str, Any]) -> None:
    """Require the standard prediction field to be a finite tensor."""
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor):
        raise PredictorError("prediction must be a tensor")
    if not bool(torch.isfinite(prediction).all()):
        raise PredictorError("prediction tensor must contain only finite values")


__all__ = [
    "TensorOutput",
    "validate_tensor_prediction",
]
