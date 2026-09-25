"""Importable text, model, metric, and prediction components for Essay Scoring."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dsio.data.store import SignalStore

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


def pad_essays(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty essay batch")
    sequences = [torch.as_tensor(item["x"], dtype=torch.long) for item in items]
    if any(value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0 for value in sequences):
        raise ValueError("essay x values must have shape [tokens, 2]")
    result: dict[str, Any] = {
        "sample_id": [str(item["sample_id"]) for item in items],
        "x": pad_sequence(sequences, batch_first=True, padding_value=0),
    }
    if all("y" in item for item in items):
        result["y"] = torch.stack([torch.as_tensor(item["y"], dtype=torch.long) for item in items])
    return result


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("QWK requires equally shaped, non-empty one-dimensional arrays")
    if truth.dtype.kind not in "iub" or predicted.dtype.kind not in "iub":
        raise ValueError("QWK ratings must be integers")
    if np.any((truth < 1) | (truth > 6)) or np.any((predicted < 1) | (predicted > 6)):
        raise ValueError("QWK ratings must be in [1, 6]")
    confusion = np.zeros((6, 6), dtype=np.float64)
    np.add.at(confusion, (truth.astype(int) - 1, predicted.astype(int) - 1), 1)
    weights = ((np.arange(6)[:, None] - np.arange(6)[None, :]) / 5.0) ** 2
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / truth.size
    denominator = float(np.sum(weights * expected))
    if denominator == 0:
        raise ValueError("QWK is undefined when expected weighted disagreement is zero")
    return 1.0 - float(np.sum(weights * confusion)) / denominator


class EssaySamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store, self.sample_ids = store, tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        result: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], dtype=np.int64, copy=True)),
        }
        if "target" in sample["attrs"]:
            result["y"] = torch.tensor(int(sample["attrs"]["target"]) - 1)
        return result


def essay_samples(
    store: SignalStore, examples: object, sample_ids: Sequence[str]
) -> Dataset[Mapping[str, Any]]:
    del examples
    return EssaySamples(store, sample_ids)


class EssayRegressor(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.output = nn.Linear(embed_dim, 6)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[2] != 2:
            raise ValueError(f"expected [batch, tokens, 2], got {tuple(x.shape)}")
        tokens = x[:, :, 0].long()
        mask = x[:, :, 1].bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every essay must contain at least one unpadded token")
        embedded = self.embedding(tokens)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.output(pooled)


class EssayObjective(nn.Module):
    def forward(
        self, model: nn.Module, batch: Mapping[str, Any], stage: str
    ) -> Mapping[str, Tensor]:
        del stage
        logits = model(batch["x"])
        return {"loss": F.cross_entropy(logits, batch["y"].long())}


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
