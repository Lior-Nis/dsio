"""Named SSL components owned by the reference consumer project."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import ComponentConfig
from dsio.experimental.model.components import NTXent
from reference_projects.supervised.components import (
    TimeMajorToChannelFirst as TimeMajorToChannelFirst,
)

DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "layout": "channel_first"},
    },
}


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
    prediction = output.get("prediction")
    if not isinstance(prediction, Tensor) or not bool(torch.isfinite(prediction).all()):
        raise ValueError("embedding norm prediction must be a finite tensor")
    if prediction.ndim != 2 or prediction.shape[1] != 1:
        raise ValueError("embedding norm prediction must have shape [batch, 1]")
    if bool((prediction < 0).any()):
        raise ValueError("embedding norm prediction must be non-negative")
