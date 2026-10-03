"""Dense multi-target components for the Parkinson's consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.config.components import ComponentConfig
from dsio.data.store import SignalStore

FEATURE_COUNT = 3
TARGET_COUNT = 3
CHANNEL_COUNT = FEATURE_COUNT + TARGET_COUNT + 1


def normalize_signal(values: np.ndarray) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32)
    mean = signal.mean(axis=0, keepdims=True)
    scale = signal.std(axis=0, keepdims=True)
    return (signal - mean) / np.maximum(scale, 1e-6)


COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0.0, "y": 0.0, "mask": False},
        "fixed_fields": [],
        "emit_mask": False,
    },
}
INPUT_COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0.0},
        "fixed_fields": [],
        "emit_mask": False,
    },
}


class FogWindows(Dataset[Mapping[str, Any]]):
    def __init__(
        self,
        store: SignalStore,
        sample_ids: Sequence[str],
        *,
        include_targets: bool = True,
    ) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        data = np.asarray(sample["data"], dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != CHANNEL_COUNT:
            raise ValueError(
                f"FoG sample {sample['sample_id']!r} must have shape "
                f"[points, {CHANNEL_COUNT}], got {data.shape}"
            )
        item: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(normalize_signal(data[:, :FEATURE_COUNT])),
        }
        if not self.include_targets:
            return item
        labels = data[:, FEATURE_COUNT:6]
        valid = data[:, -1]
        if not bool(np.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError(
                f"FoG sample {sample['sample_id']!r} labels must be finite binary values"
            )
        if not bool(np.isfinite(valid).all()) or not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError(
                f"FoG sample {sample['sample_id']!r} mask must contain only zero or one"
            )
        item["y"] = torch.from_numpy(np.array(labels, copy=True))
        item["mask"] = torch.from_numpy(np.array(valid, dtype=bool, copy=True))
        return item


def fog_windows(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return FogWindows(store, sample_ids)


def fog_inputs(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return FogWindows(store, sample_ids, include_targets=False)


class FogDetector(nn.Module):
    def __init__(self, features: int = FEATURE_COUNT, hidden: int = 16) -> None:
        super().__init__()
        self.features = features
        self.network = nn.Sequential(
            nn.Conv1d(features, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, TARGET_COUNT, kernel_size=5, padding=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != self.features:
            raise ValueError(f"expected [batch, points, {self.features}], got {tuple(x.shape)}")
        return self.network(x.float().transpose(1, 2)).transpose(1, 2)


class FogObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        logits = model(batch["x"])
        mask = batch["mask"].bool()
        if not bool(mask.any()):
            raise ValueError("a dense-prediction batch must contain at least one valid point")
        loss = F.binary_cross_entropy_with_logits(logits[mask], batch["y"][mask].float())
        return {"loss": loss}


class FogOutput(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        probability = torch.sigmoid(logits)
        return {"prediction": (probability >= 0.5).long(), "probability": probability}


def validate_fog_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    probability = output.get("probability")
    if not isinstance(prediction, Tensor) or prediction.ndim != 3:
        raise ValueError("FoG prediction must have shape [batch, points, classes]")
    if not isinstance(probability, Tensor) or probability.ndim != 3:
        raise ValueError("FoG probability must have shape [batch, points, classes]")
    if prediction.shape != probability.shape or probability.shape[2] != TARGET_COUNT:
        raise ValueError(f"FoG probability must have {TARGET_COUNT} classes")
    if not bool(((prediction == 0) | (prediction == 1)).all()):
        raise ValueError("FoG prediction must be binary")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("FoG probability must be finite")
    if not bool(((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("FoG probability must be between zero and one")
    if not torch.equal(prediction, (probability >= 0.5).long()):
        raise ValueError("FoG prediction must equal probability thresholded at 0.5")
