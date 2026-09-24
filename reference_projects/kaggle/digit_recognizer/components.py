"""Importable Digit Recognizer representation and classifier components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.store import SignalStore


class UnlabelledDigitSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], copy=True)).float() / 255.0,
        }


def unlabelled_digit_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return UnlabelledDigitSamples(store, sample_ids)


class LabelledDigitSamples(UnlabelledDigitSamples):
    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], copy=True)).float() / 255.0,
            "y": torch.tensor(int(sample["attrs"]["target"]), dtype=torch.int64),
        }


def labelled_digit_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return LabelledDigitSamples(store, sample_ids)


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


class ScalePixels(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (28, 28):
            raise ValueError(f"expected [batch, 28, 28], got {tuple(x.shape)}")
        return x.float() / 255.0


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
