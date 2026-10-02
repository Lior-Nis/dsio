"""Packed multimodal components for the project-owned CMI experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dsio.config.components import ComponentConfig
from reference_projects.kaggle.child_mind.data import (
    PACKED_FEATURES,
    SENSOR_FEATURES,
    SENSOR_MASK_SLICE,
    SENSOR_PRESENT_INDEX,
    SENSOR_VALUE_SLICE,
    TABULAR_FEATURES,
    TABULAR_MASK_SLICE,
    TABULAR_VALUE_SLICE,
)

CLASSES = 4


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
    },
}


class CmiFusionClassifier(nn.Module):
    def __init__(
        self,
        *,
        tabular_center: Sequence[float],
        tabular_scale: Sequence[float],
        sensor_center: Sequence[float],
        sensor_scale: Sequence[float],
        use_sensor: bool = True,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if isinstance(use_sensor, bool) is False:
            raise ValueError("use_sensor must be bool")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.register_buffer(
            "tabular_center", _vector(tabular_center, TABULAR_FEATURES, "tabular_center")
        )
        self.register_buffer(
            "tabular_scale", _scale(tabular_scale, TABULAR_FEATURES, "tabular_scale")
        )
        self.register_buffer(
            "sensor_center", _vector(sensor_center, SENSOR_FEATURES, "sensor_center")
        )
        self.register_buffer("sensor_scale", _scale(sensor_scale, SENSOR_FEATURES, "sensor_scale"))
        self.use_sensor = use_sensor
        self.tabular_encoder = nn.Sequential(nn.Linear(TABULAR_FEATURES * 2, hidden), nn.ReLU())
        self.sensor_encoder = nn.Sequential(nn.Linear(SENSOR_FEATURES * 2, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, CLASSES),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 2 or x.shape[1] != PACKED_FEATURES:
            raise ValueError(
                f"CMI input must have shape [batch, {PACKED_FEATURES}] or "
                f"[batch, 1, {PACKED_FEATURES}], got {tuple(x.shape)}"
            )
        x = x.float()
        tabular_mask = x[:, TABULAR_MASK_SLICE]
        sensor_mask = x[:, SENSOR_MASK_SLICE]
        present = x[:, SENSOR_PRESENT_INDEX : SENSOR_PRESENT_INDEX + 1]
        _binary_mask(tabular_mask, "tabular mask")
        _binary_mask(sensor_mask, "sensor mask")
        _binary_mask(present, "sensor presence")
        if bool(((present == 0) & (sensor_mask != 0).any(dim=1, keepdim=True)).any()):
            raise ValueError("absent sensor modality cannot contain observed sensor features")

        tabular = _normalize_observed(
            x[:, TABULAR_VALUE_SLICE],
            tabular_mask,
            self.tabular_center,
            self.tabular_scale,
            "tabular",
        )
        tabular_encoded = self.tabular_encoder(torch.cat([tabular, tabular_mask], 1))
        sensor = _normalize_observed(
            x[:, SENSOR_VALUE_SLICE],
            sensor_mask,
            self.sensor_center,
            self.sensor_scale,
            "sensor",
        )
        sensor_encoded = self.sensor_encoder(torch.cat([sensor, sensor_mask], 1))
        if self.use_sensor:
            sensor_encoded = sensor_encoded * present
        else:
            sensor_encoded = torch.zeros_like(sensor_encoded)
            present = torch.zeros_like(present)
        return self.head(torch.cat([tabular_encoded, sensor_encoded, present], 1))


class CmiObjective(nn.Module):
    def __init__(self, class_weights: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("class_weights", _scale(class_weights, CLASSES, "class_weights"))

    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        logits = model(batch["x"])
        target = batch["y"].long().reshape(-1)
        return {
            "loss": F.cross_entropy(logits, target, weight=self.class_weights),
            "accuracy": (logits.argmax(dim=1) == target).float().mean(),
        }


class CmiPrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        probability = torch.softmax(logits, dim=1)
        return {"prediction": probability.argmax(dim=1), "probability": probability}


def validate_cmi_prediction(output: Mapping[str, Any]) -> None:
    prediction, probability = output.get("prediction"), output.get("probability")
    if not isinstance(prediction, Tensor) or prediction.ndim != 1:
        raise ValueError("CMI prediction must have shape [batch]")
    if prediction.shape[0] == 0:
        raise ValueError("CMI prediction batch must be non-empty")
    if prediction.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("CMI prediction must have an integer dtype")
    if not isinstance(probability, Tensor) or probability.ndim != 2:
        raise ValueError("CMI probability must have shape [batch, classes]")
    if probability.shape != (prediction.shape[0], CLASSES):
        raise ValueError(f"CMI probability must have {CLASSES} classes")
    if not bool(((prediction >= 0) & (prediction < CLASSES)).all()):
        raise ValueError(f"CMI prediction must be in [0, {CLASSES - 1}]")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("CMI probability must be finite")
    if not bool(((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("CMI probability values must be in [0, 1]")
    if not torch.allclose(probability.sum(dim=1), torch.ones_like(probability[:, 0])):
        raise ValueError("CMI probabilities must sum to one")
    if not torch.equal(prediction, probability.argmax(dim=1)):
        raise ValueError("CMI prediction must be the probability argmax")


def ablate_sensor(x: Tensor) -> Tensor:
    if x.shape[-1] != PACKED_FEATURES:
        raise ValueError(f"CMI ablation expects {PACKED_FEATURES} packed features")
    result = x.clone()
    result[..., SENSOR_VALUE_SLICE] = 0
    result[..., SENSOR_MASK_SLICE] = 0
    result[..., SENSOR_PRESENT_INDEX] = 0
    return result


def _vector(values: Sequence[float], length: int, name: str) -> Tensor:
    result = torch.as_tensor(list(values), dtype=torch.float32)
    if result.shape != (length,) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _scale(values: Sequence[float], length: int, name: str) -> Tensor:
    result = _vector(values, length, name)
    if not bool((result > 0).all()):
        raise ValueError(f"{name} values must be positive")
    return result


def _binary_mask(values: Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()) or not bool(((values == 0) | (values == 1)).all()):
        raise ValueError(f"CMI {name} must contain only zero or one")


def _normalize_observed(
    values: Tensor,
    mask: Tensor,
    center: Tensor,
    scale: Tensor,
    name: str,
) -> Tensor:
    observed = mask.bool()
    observed_values = torch.where(observed, values, torch.zeros_like(values))
    if not bool(torch.isfinite(observed_values).all()):
        raise ValueError(f"CMI observed {name} values must be finite")
    safe_values = torch.where(observed, values, center)
    normalized = (safe_values - center) / scale
    return torch.where(observed, normalized, torch.zeros_like(normalized))


__all__ = [
    "CLASSES",
    "DATASET",
    "CmiFusionClassifier",
    "CmiObjective",
    "CmiPrediction",
    "ablate_sensor",
    "validate_cmi_prediction",
]
