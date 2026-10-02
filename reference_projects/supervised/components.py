"""Small importable components owned by the reference consumer project."""

from __future__ import annotations

from pathlib import Path

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

# Predictor inputs: raw time-major signals, exactly as the exported model receives them.
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data"}},
}

# Evaluation inputs (raw time-major, as the predictor receives them) and targets,
# assembled through the training collation.
EVALUATION: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data"},
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


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs:RegressionOutput",
    "parameters": {"shape": [1]},
}

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
