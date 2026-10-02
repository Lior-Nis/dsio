"""Small importable components owned by the reference consumer project."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor, nn

from dsio.config.components import ComponentConfig
from dsio.data.store import SignalStore

DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "layout": "channel_first"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}


class TimeMajorToChannelFirst(nn.Module):
    """Validate raw signal shape and prepare the shared channel-first model layout."""

    def __init__(self, *, channels: int, time: int) -> None:
        super().__init__()
        for name, extent in (("channels", channels), ("time", time)):
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise ValueError(f"{name} extent must be a positive integer, got {extent!r}")
        self.channels = channels
        self.time = time

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time, channels], got shape {tuple(x.shape)}")
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


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "metrics": {"mae": {"reference": "torch.nn:L1Loss"}},
    },
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
