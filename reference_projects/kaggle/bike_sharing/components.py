"""Importable Bike Sharing training and inference components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.store import SignalStore

FEATURES = 9


class HourlySamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], dtype=np.float32, copy=True)),
            "y": torch.tensor([float(sample["attrs"]["target"])], dtype=torch.float32),
        }


def hourly_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return HourlySamples(store, sample_ids)


class DemandRegressor(nn.Module):
    def __init__(self, mean: Sequence[float], scale: Sequence[float]) -> None:
        super().__init__()
        if len(mean) != FEATURES or len(scale) != FEATURES:
            raise ValueError(f"mean and scale must each contain {FEATURES} values")
        self.mean: Tensor
        self.scale: Tensor
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).reshape(1, 1, -1))
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(FEATURES, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, FEATURES):
            raise ValueError(f"expected [batch, 1, {FEATURES}], got {tuple(x.shape)}")
        return F.softplus(self.network((x.float() - self.mean) / self.scale))


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
