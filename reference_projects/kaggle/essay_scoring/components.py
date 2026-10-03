"""Importable text, model, metric, and prediction components for Essay Scoring."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import ComponentConfig

MAX_TOKENS = 512
VOCAB_SIZE = 4096
_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[int]:
    """Map text to stable, dependency-free hash buckets; zero remains padding."""
    values = []
    for token in _TOKEN.findall(text.casefold())[:MAX_TOKENS]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        values.append(int.from_bytes(digest, "little") % (VOCAB_SIZE - 1) + 1)
    if not values:
        raise ValueError("an essay must produce at least one token")
    return values


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "int64"},
        "y": {
            "from": "attribute",
            "attribute": "target",
            "dtype": "int64",
            "offset": -1,
        },
    },
}
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data", "dtype": "int64"}},
}
COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0},
        "fixed_fields": ["y"],
        "emit_mask": False,
    },
}
MODEL: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:Chain",
    "parameters": {
        "backbone": {
            "reference": "dsio.experimental.model.tokens:EmbeddingEncoder",
            "parameters": {"vocab_size": VOCAB_SIZE, "embed_dim": 32},
        },
        "head": {
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 32, "out_features": 6},
        },
    },
}
OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:CrossEntropyLoss"},
        "target_dtype": "int64",
    },
}


class OrdinalPrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        probabilities = torch.softmax(logits, dim=1)
        return {"prediction": torch.argmax(probabilities, dim=1) + 1, "score": probabilities}


def validate_ordinal_prediction(output: Mapping[str, Any]) -> None:
    prediction, score = output.get("prediction"), output.get("score")
    if not isinstance(prediction, Tensor) or prediction.ndim != 1:
        raise ValueError("ordinal prediction must have shape [batch]")
    if not bool(torch.all((prediction >= 1) & (prediction <= 6))):
        raise ValueError("ordinal predictions must be in [1, 6]")
    if not isinstance(score, Tensor) or score.shape != (prediction.shape[0], 6):
        raise ValueError("ordinal score must have shape [batch, 6]")
    if not bool(torch.isfinite(score).all()) or not torch.allclose(
        score.sum(dim=1), torch.ones_like(score[:, 0]), atol=1e-6
    ):
        raise ValueError("ordinal scores must be finite probability distributions")
