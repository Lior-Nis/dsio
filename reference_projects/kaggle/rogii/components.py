"""Variable-length dense-regression components for the ROGII consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dsio.data.store import SignalStore

FEATURES = 13
TARGET_SCALE = 20_000.0


def well_arrays(well: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tail = well["tail"]
    origin = tail[0]
    type_tvt = np.asarray(well["typewell"]["TVT"], dtype=np.float64)
    type_gr = np.asarray(well["typewell"]["GR"], dtype=np.float64)
    rows = []
    targets = []
    for item in tail:
        gr = item["GR"]
        rows.append(
            [
                (item["MD"] - origin["MD"]) / 10_000,
                (item["X"] - origin["X"]) / 10_000,
                (item["Y"] - origin["Y"]) / 10_000,
                (item["Z"] - origin["Z"]) / 10_000,
                0.0 if gr is None else gr / 200,
                float(gr is None),
                well["last_known_tvt"] / TARGET_SCALE,
                well["last_tvt_slope"],
                float(type_tvt.min()) / TARGET_SCALE,
                float(type_tvt.max()) / TARGET_SCALE,
                float(type_gr.mean()) / 200,
                float(type_gr.std()) / 200,
                1.0,
            ]
        )
        targets.append(0.0 if "TVT" not in item else item["TVT"] / TARGET_SCALE)
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def pad_wells(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty well batch")
    x = [torch.as_tensor(item["x"], dtype=torch.float32) for item in items]
    y = [torch.as_tensor(item["y"], dtype=torch.float32) for item in items]
    if any(value.ndim != 2 or value.shape[1] != FEATURES for value in x):
        raise ValueError(f"well x values must have shape [points, {FEATURES}]")
    return {
        "sample_id": [str(item["sample_id"]) for item in items],
        "x": pad_sequence(x, batch_first=True),
        "y": pad_sequence(y, batch_first=True),
    }


class WellSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        data = np.asarray(sample["data"], dtype=np.float32)
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(data[:, :FEATURES], copy=True)),
            "y": torch.from_numpy(np.array(data[:, FEATURES], copy=True)),
        }


def well_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return WellSamples(store, sample_ids)


class TvtRegressor(nn.Module):
    def __init__(self, features: int = FEATURES, hidden: int = 32) -> None:
        super().__init__()
        self.features = features
        self.residual = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        output = self.residual[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.features:
            raise ValueError(f"expected [batch, points, {self.features}], got {tuple(x.shape)}")
        valid = x[:, :, -1]
        baseline = x[:, :, 6]
        correction = 0.01 * torch.tanh(self.residual(x.float()).squeeze(-1))
        return (baseline + correction) * valid


class TvtObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        prediction = model(batch["x"])
        valid = batch["x"][:, :, -1].bool()
        loss = F.mse_loss(prediction[valid], batch["y"][valid].float())
        return {"loss": loss, "rmse": torch.sqrt(loss.detach()) * TARGET_SCALE}


class TvtOutput(nn.Module):
    def forward(self, normalized: Tensor) -> Mapping[str, Tensor]:
        return {"prediction": normalized * TARGET_SCALE}


def validate_tvt_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor) or prediction.ndim != 2:
        raise ValueError("TVT prediction must have shape [batch, points]")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("TVT prediction must be finite")
