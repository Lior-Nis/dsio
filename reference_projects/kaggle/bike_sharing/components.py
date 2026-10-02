"""Importable Bike Sharing training and inference components."""

from __future__ import annotations

from dsio.config.components import ComponentConfig

FEATURES = 9


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}

# Predictor inputs, exactly as the exported model receives them.
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
    },
}


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "metrics": {"mae": {"reference": "torch.nn:L1Loss"}},
    },
}


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs:RegressionOutput",
    "parameters": {"shape": [1], "non_negative": True},
}
