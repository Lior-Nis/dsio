"""Small importable components owned by the reference consumer project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from dsio.data.store import SignalStore


class RegressionSamples(Dataset[Mapping[str, Any]]):
    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        return {
            "sample_id": sample["sample_id"],
            "x": torch.from_numpy(np.array(sample["data"], copy=True)),
            "y": torch.tensor([float(sample["attrs"]["target"])], dtype=torch.float32),
        }


def regression_samples(
    store: SignalStore,
    examples: object,
    sample_ids: Sequence[str],
) -> Dataset[Mapping[str, Any]]:
    del examples
    return RegressionSamples(store, sample_ids)


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x.float())


class RegressionObjective(nn.Module):
    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, Tensor]:
        del stage
        prediction = model(batch["x"])
        target = batch["y"].float()
        return {
            "loss": F.mse_loss(prediction, target),
            "mae": F.l1_loss(prediction, target),
        }


def build_synthetic_store(path: Path, seed: int) -> SignalStore:
    generator = np.random.default_rng(seed)
    with SignalStore.builder(path, channels=1, dtype="float32") as builder:
        for index in range(8):
            signal = generator.normal(loc=index / 4, scale=0.05, size=(4, 1)).astype(np.float32)
            target = float(signal.sum(dtype=np.float64) * 0.4 + 0.25)
            builder.add(
                f"sample-{index}",
                signal,
                group=f"group-{index}",
                attrs={"target": target},
            )
    return SignalStore(path)


def evaluation_arrays(
    store_path: str,
    sample_ids: Sequence[str],
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    store = SignalStore(store_path)
    samples = [store.read_sample(sample_id) for sample_id in sample_ids]
    inputs = {
        "sample_id": np.asarray([sample["sample_id"] for sample in samples], dtype=np.str_),
        "x": np.stack([np.array(sample["data"], copy=True) for sample in samples]).astype(
            np.float32
        ),
    }
    targets = np.asarray(
        [[float(sample["attrs"]["target"])] for sample in samples],
        dtype=np.float32,
    )
    return inputs, targets
