"""Importable Store Sales training and inference components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.store import SignalStore
from reference_projects.kaggle.store_sales.data import CONTEXT_DAYS, HORIZON_DAYS

FEATURES = 5


class ForecastSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        target = np.asarray(sample["attrs"]["target"], dtype=np.float32)
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], dtype=np.float32, copy=True)),
            "y": torch.from_numpy(np.log1p(target)),
        }


def forecast_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return ForecastSamples(store, sample_ids)


class ForecastRegressor(nn.Module):
    def __init__(
        self,
        context_days: int = CONTEXT_DAYS,
        horizon_days: int = HORIZON_DAYS,
        features: int = FEATURES,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.context_days = context_days
        self.horizon_days = horizon_days
        self.features = features
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear((context_days + horizon_days) * features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon_days),
            nn.Softplus(),
        )

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.context_days + self.horizon_days, self.features)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected [batch, {expected[0]}, {expected[1]}], got {tuple(x.shape)}"
            )
        return self.network(x.float())


class ForecastObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        prediction = model(batch["x"])
        loss = F.mse_loss(prediction, batch["y"].float())
        return {"loss": loss, "rmsle": torch.sqrt(loss.detach())}


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
