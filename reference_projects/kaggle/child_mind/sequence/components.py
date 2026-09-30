"""Identity-preserving raw-sequence dataset, temporal model, and objective."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchmetrics import MeanMetric

from dsio.data.adapters import SignalExamples
from dsio.data.store import SignalStore
from reference_projects.kaggle.child_mind.components import CLASSES
from reference_projects.kaggle.child_mind.data import TABULAR_FEATURES


class CmiSequenceWindows(Dataset[Mapping[str, Any]]):
    """Resolve governed window identities into one Predictor-compatible input tensor."""

    def __init__(
        self,
        store: SignalStore,
        examples: SignalExamples,
        sample_ids: Sequence[str],
    ) -> None:
        self.store = store
        self.examples = examples
        positions = {
            examples.index.sample_id(position): position for position in range(len(examples))
        }
        try:
            self.positions = tuple(positions[str(sample_id)] for sample_id in sample_ids)
        except KeyError as error:
            raise ValueError(f"unknown CMI sequence sample identity {error.args[0]!r}") from None
        self.sample_ids = tuple(str(value) for value in sample_ids)
        counts = Counter(int(examples.index.entity_codes[position]) for position in self.positions)
        participants = len(counts)
        self.weights = {
            code: len(self.positions) / (participants * count) for code, count in counts.items()
        }

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> Mapping[str, Any]:
        position = self.positions[item]
        code = int(self.examples.index.entity_codes[position])
        entity = self.store.entities[code]
        start = int(self.examples.index.starts[position])
        signal = np.asarray(
            self.store.read(start, self.examples.index.spec.length), dtype=np.float32
        )
        present = bool(entity.attrs["sensor_present"])
        tabular = np.asarray(entity.attrs["tabular"], dtype=np.float32)
        x = np.concatenate(
            (
                tabular,
                signal.reshape(-1),
                np.asarray([float(present)], dtype=np.float32),
            )
        )
        return {
            "sample_id": self.sample_ids[item],
            "participant_id": entity.group,
            "x": torch.from_numpy(x),
            "y": torch.tensor(int(entity.attrs["label"]), dtype=torch.long),
            "sample_weight": torch.tensor(self.weights[code], dtype=torch.float32),
            "sensor_present": present,
        }


def sequence_windows(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    if not isinstance(examples, SignalExamples):
        raise TypeError("CMI sequence windows require SignalExamples")
    return CmiSequenceWindows(store, examples, sample_ids)


class CmiSequenceClassifier(nn.Module):
    """Fuse masked tabular values with a strided temporal encoder."""

    def __init__(
        self,
        *,
        window_length: int,
        sensor_features: int,
        tabular_center: Sequence[float],
        tabular_scale: Sequence[float],
        hidden: int = 32,
    ) -> None:
        super().__init__()
        for name, value in (
            ("window_length", window_length),
            ("sensor_features", sensor_features),
            ("hidden", hidden),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.window_length = window_length
        self.sensor_features = sensor_features
        self.input_features = TABULAR_FEATURES * 2 + window_length * sensor_features + 1
        self.tabular_center: Tensor
        self.register_buffer(
            "tabular_center", _vector(tabular_center, TABULAR_FEATURES, "tabular_center")
        )
        self.tabular_scale: Tensor
        self.register_buffer(
            "tabular_scale", _scale(tabular_scale, TABULAR_FEATURES, "tabular_scale")
        )
        self.tabular_encoder = nn.Sequential(nn.Linear(TABULAR_FEATURES * 2, hidden), nn.GELU())
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(sensor_features * 2, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, CLASSES),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError(
                f"CMI sequence input must have shape [batch, {self.input_features}], "
                f"got {tuple(x.shape)}"
            )
        x = x.float()
        tabular_values = x[:, :TABULAR_FEATURES]
        tabular_mask = x[:, TABULAR_FEATURES : TABULAR_FEATURES * 2]
        signal_start = TABULAR_FEATURES * 2
        signal = x[:, signal_start:-1].reshape(-1, self.window_length, self.sensor_features)
        present = x[:, -1:]
        finite = torch.isfinite(signal)
        absent = present[:, None, :].eq(0)
        if bool((absent & finite & signal.ne(0)).any()):
            raise ValueError("absent sensor modality must contain only zero placeholders")
        observed = (finite & ~absent).to(x.dtype)
        _binary(tabular_mask, "tabular mask")
        _binary(observed, "sensor mask")
        _binary(present, "sensor presence")

        safe_tabular = torch.where(tabular_mask.bool(), tabular_values, self.tabular_center)
        normalized_tabular = (safe_tabular - self.tabular_center) / self.tabular_scale
        normalized_tabular = normalized_tabular * tabular_mask
        tabular_encoded = self.tabular_encoder(torch.cat((normalized_tabular, tabular_mask), dim=1))

        normalized_signal = _normalize_window(signal, observed)
        temporal = self.temporal_encoder(
            torch.cat((normalized_signal, observed), dim=2).transpose(1, 2)
        )
        temporal_encoded = temporal.mean(dim=2) * present
        return self.head(torch.cat((tabular_encoded, temporal_encoded, present), dim=1))


class CmiSequenceObjective(nn.Module):
    def __init__(self, class_weights: Sequence[float]) -> None:
        super().__init__()
        self.class_weights: Tensor
        self.register_buffer("class_weights", _scale(class_weights, CLASSES, "class_weights"))
        self.metrics = nn.ModuleDict(
            {f"{stage}_accuracy": MeanMetric() for stage in ("train", "validate", "test")}
        )

    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor | MeanMetric]:
        logits = model(batch["x"])
        target = batch["y"].long().reshape(-1)
        weights = batch["sample_weight"].float().reshape(-1)
        if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
            raise ValueError("CMI sequence sample weights must be finite and positive")
        losses = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none")
        try:
            accuracy = self.metrics[f"{stage}_accuracy"]
        except KeyError:
            raise ValueError(f"unsupported CMI sequence objective stage {stage!r}") from None
        accuracy.update((logits.argmax(dim=1) == target).float(), weight=weights)
        return {
            # Dataset weights have a fixed split-wide mean of one. A plain mean
            # therefore preserves participant balancing across microbatch boundaries.
            "loss": (losses * weights).mean(),
            "accuracy": accuracy,
        }


class CmiSequencePrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        probability = torch.softmax(logits, dim=1)
        return {"prediction": probability.argmax(dim=1), "probability": probability}


def validate_sequence_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    probability = output.get("probability")
    if not isinstance(prediction, Tensor) or prediction.ndim != 1:
        raise ValueError("CMI sequence prediction must have shape [batch]")
    if not isinstance(probability, Tensor) or probability.shape != (len(prediction), CLASSES):
        raise ValueError(f"CMI sequence probability must have shape [batch, {CLASSES}]")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("CMI sequence probability must be finite")
    if not torch.allclose(probability.sum(dim=1), torch.ones_like(probability[:, 0])):
        raise ValueError("CMI sequence probabilities must sum to one")
    if not torch.equal(prediction, probability.argmax(dim=1)):
        raise ValueError("CMI sequence prediction must be the probability argmax")


def ablate_sequence_sensor(x: Tensor, *, window_length: int, sensor_features: int) -> Tensor:
    expected = TABULAR_FEATURES * 2 + window_length * sensor_features + 1
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"CMI sequence ablation expects [batch, {expected}]")
    result = x.clone()
    result[:, TABULAR_FEATURES * 2 :] = 0
    return result


def _normalize_window(signal: Tensor, observed: Tensor) -> Tensor:
    mask = observed.bool()
    safe = torch.where(mask, signal, torch.zeros_like(signal))
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("CMI observed sensor values must be finite")
    count = observed.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = safe.sum(dim=1, keepdim=True) / count
    variance = (torch.where(mask, signal - mean, torch.zeros_like(signal)).square()).sum(
        dim=1, keepdim=True
    ) / count
    scale = variance.sqrt().clamp_min(1e-6)
    return torch.where(mask, (signal - mean) / scale, torch.zeros_like(signal))


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


def _binary(values: Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()) or not bool(((values == 0) | (values == 1)).all()):
        raise ValueError(f"CMI {name} must contain only zero or one")


__all__ = [
    "CmiSequenceClassifier",
    "CmiSequenceObjective",
    "CmiSequencePrediction",
    "CmiSequenceWindows",
    "ablate_sequence_sensor",
    "sequence_windows",
    "validate_sequence_prediction",
]
