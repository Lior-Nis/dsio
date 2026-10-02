"""Importable Digit Recognizer representation and classifier components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dsio.config.components import ComponentConfig

UNLABELLED_DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32", "divide": 255.0},
    },
}

LABELLED_DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32", "divide": 255.0},
        "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
    },
}


def _encoder() -> nn.Sequential:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 16))


class DigitAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _encoder()
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 784), nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x.float())

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encode(x)).reshape(-1, 28, 28)


class ReconstructionObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        return {"loss": F.mse_loss(model(batch["x"]), batch["x"].float())}


class FrozenDigitClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _encoder()
        self.classifier = nn.Linear(16, 10)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            encoded = self.encoder(x.float())
        return self.classifier(encoded)


class ClassificationObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        return {"loss": F.cross_entropy(model(batch["x"]), batch["y"].long())}


class DigitPrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != 10 or not bool(torch.isfinite(logits).all()):
            raise ValueError("digit logits must be a finite [batch, 10] tensor")
        return {"prediction": torch.argmax(logits, dim=1)}


def validate_digit_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if (
        not isinstance(prediction, Tensor)
        or prediction.ndim != 1
        or prediction.dtype != torch.int64
    ):
        raise ValueError("digit prediction must be an int64 [batch] tensor")
    if not bool(torch.all((prediction >= 0) & (prediction <= 9))):
        raise ValueError("digit predictions must be in [0, 9]")
