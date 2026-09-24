"""Importable Titanic model, dataset, and prediction components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.store import SignalStore

FEATURES = 10


class PassengerSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], dtype=np.float32, copy=True)),
            "y": torch.tensor([float(sample["attrs"]["target"])], dtype=torch.float32),
        }


def passenger_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return PassengerSamples(store, sample_ids)


class PassengerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(FEATURES, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, FEATURES):
            raise ValueError(f"expected [batch, 1, {FEATURES}], got {tuple(x.shape)}")
        return self.network(x.float())


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
