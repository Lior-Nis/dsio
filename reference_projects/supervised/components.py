"""Small importable components owned by the reference consumer project."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dsio.config.components import ComponentConfig
from dsio.data.store import SignalStore

DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "layout": "channel_first"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}

COLLATOR: ComponentConfig = {
    "reference": "dsio.data.loading.collation:IdentityCollator",
    "parameters": {},
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


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.regression:RegressionOutput",
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
