"""Importable Store Sales training and inference components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import ComponentConfig
from reference_projects.kaggle.store_sales.data import HORIZON_DAYS

FEATURES = 5


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "log1p": True},
    },
}


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "metrics": {
            "rmsle": {"reference": "dsio.experimental.model.objectives:RootMeanSquaredError"}
        },
    },
}


class ForecastOutput(nn.Module):
    def forward(self, log_prediction: Tensor) -> Mapping[str, Tensor]:
        return {
            "prediction": torch.expm1(log_prediction).clamp_min(0),
            "log_prediction": log_prediction,
        }


def validate_forecast(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    log_prediction = output.get("log_prediction")
    for name, value in (("prediction", prediction), ("log_prediction", log_prediction)):
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != HORIZON_DAYS:
            raise ValueError(f"{name} must be a [batch, {HORIZON_DAYS}] tensor")
        if not bool(torch.isfinite(value).all()) or not bool(torch.all(value >= 0)):
            raise ValueError(f"{name} must be finite and non-negative")
