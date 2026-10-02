"""Importable Bike Sharing training and inference components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dsio.config.components import ComponentConfig

FEATURES = 9


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}


class DemandObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        prediction = model(batch["x"])
        return {
            "loss": F.mse_loss(prediction, batch["y"].float()),
            "mae": F.l1_loss(prediction, batch["y"].float()),
        }


class NonNegativeOutput(nn.Module):
    def forward(self, prediction: Tensor) -> Mapping[str, Tensor]:
        return {"prediction": prediction}


def validate_nonnegative_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor) or prediction.ndim != 2 or prediction.shape[1] != 1:
        raise ValueError("demand prediction must be a [batch, 1] tensor")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.all(prediction >= 0)):
        raise ValueError("demand predictions must be finite and non-negative")
