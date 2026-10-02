"""Importable Titanic model, dataset, and prediction components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dsio.config.components import ComponentConfig

FEATURES = 10


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}


class PassengerObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        return {"loss": F.binary_cross_entropy_with_logits(model(batch["x"]), batch["y"].float())}


class BinaryPrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        score = torch.sigmoid(logits).reshape(-1)
        return {"prediction": (score >= 0.5).to(torch.int64), "score": score}


def validate_binary_prediction(output: Mapping[str, Any]) -> None:
    prediction, score = output.get("prediction"), output.get("score")
    if not isinstance(prediction, Tensor) or not isinstance(score, Tensor):
        raise ValueError("binary prediction requires tensor prediction and score fields")
    if prediction.shape != score.shape or prediction.ndim != 1:
        raise ValueError("binary prediction and score must have shape [batch]")
    if not bool(torch.all((prediction == 0) | (prediction == 1))):
        raise ValueError("binary predictions must be 0 or 1")
    if not bool(torch.isfinite(score).all()) or not bool(torch.all((score >= 0) & (score <= 1))):
        raise ValueError("binary scores must be finite probabilities")
