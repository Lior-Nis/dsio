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

COLLATOR: ComponentConfig = {
    "reference": "dsio.data.loading.collation:IdentityCollator",
    "parameters": {},
}

# Predictor inputs, exactly as the exported model receives them.
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
    },
}

# Evaluation uses the training item mapping. BinaryOutput exposes one label per sample,
# so the downstream task removes only the training loss's singleton target axis.
EVALUATION: ComponentConfig = DATASET


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {"loss": {"reference": "torch.nn:BCEWithLogitsLoss"}},
}


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.binary:BinaryOutput",
    "parameters": {},
}
