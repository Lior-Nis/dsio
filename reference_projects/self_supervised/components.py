"""Named SSL components owned by the reference consumer project."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.inference import validate_tensor_prediction
from dsio.model.components import NTXent


class TinyEmbedding(nn.Module):
    """A tiny deterministic embedding network for the synthetic signal."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
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
