"""Importable Titanic model, dataset, and prediction components."""

from __future__ import annotations

from dsio.config.components import ComponentConfig

FEATURES = 10


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "shape": [1]},
    },
}


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {"loss": {"reference": "torch.nn:BCEWithLogitsLoss"}},
}


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs:BinaryOutput",
    "parameters": {},
}
