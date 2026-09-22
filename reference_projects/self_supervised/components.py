"""Named SSL components owned by the reference consumer project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from dsio.data.store import SignalStore
from dsio.inference import validate_tensor_prediction
from dsio.model.components import NTXent


class UnlabelledSamples(Dataset[Mapping[str, Any]]):
    """Expose stored time-series in the channel-first shape augmentors expect."""

    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        data = np.array(sample["data"].T, copy=True)
        return {"sample_id": sample["sample_id"], "x": torch.from_numpy(data)}


def unlabelled_samples(
    store: SignalStore,
    examples: object,
    sample_ids: Sequence[str],
) -> Dataset[Mapping[str, Any]]:
    del examples
    return UnlabelledSamples(store, sample_ids)


class TimeMajorToChannelFirst(nn.Module):
    """Validate the external signal shape and match the training tensor layout."""

    def __init__(self, *, channels: int, time: int) -> None:
        super().__init__()
        for name, extent in (("channels", channels), ("time", time)):
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise ValueError(f"{name} extent must be a positive integer, got {extent!r}")
        self.channels = channels
        self.time = time

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "expected [batch, time, channels], "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.time:
            raise ValueError(
                f"expected time extent {self.time}, got {x.shape[1]} in shape {tuple(x.shape)}"
            )
        if x.shape[2] != self.channels:
            raise ValueError(
                f"expected channel extent {self.channels}, got {x.shape[2]} "
                f"in shape {tuple(x.shape)}"
            )
        return x.transpose(1, 2).contiguous()


class TinyEmbedding(nn.Module):
    """A tiny deterministic embedding network for the synthetic signal."""

    input_shape = (1, 4)

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                "TinyEmbedding expects [batch, channels, time] shape "
                f"(batch, {self.input_shape[0]}, {self.input_shape[1]}), "
                f"got {tuple(x.shape)}"
            )
        return self.network(x.float())


class ContrastiveObjective(nn.Module):
    """Apply the named NT-Xent objective to the two-view training batch."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.loss = NTXent(temperature=temperature)

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, Tensor]:
        if stage != "train":
            raise ValueError("contrastive objective is training-only")
        prediction = model(batch["x"])
        loss = self.loss(prediction, batch["y"])
        diagnostics = self.loss.diagnostics(prediction, batch["y"], batch["x"])
        return {"loss": loss, **diagnostics}


class EmbeddingNorm(nn.Module):
    """Expose one finite scalar per embedding through the shared predictor contract."""

    def forward(self, embedding: Tensor) -> Mapping[str, Tensor]:
        return {"prediction": torch.linalg.vector_norm(embedding, dim=-1, keepdim=True)}


def validate_embedding_norm(output: Mapping[str, Any]) -> None:
    """Require one finite, non-negative norm for every source sample."""
    validate_tensor_prediction(output)
    prediction = output["prediction"]
    assert isinstance(prediction, Tensor)
    if prediction.ndim != 2 or prediction.shape[1] != 1:
        raise ValueError("embedding norm prediction must have shape [batch, 1]")
    if bool((prediction < 0).any()):
        raise ValueError("embedding norm prediction must be non-negative")
