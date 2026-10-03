"""Importable Store Sales training and inference components."""

from __future__ import annotations

from dsio.config.components import ComponentConfig
from reference_projects.kaggle.store_sales.data import HORIZON_DAYS

FEATURES = 5


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "float32", "log1p": True},
    },
}

COLLATOR: ComponentConfig = {
    "reference": "dsio.data.loading.collation:IdentityCollator",
    "parameters": {},
}

# Predictor inputs, exactly as the exported model receives them.
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data", "dtype": "float32"}},
}


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "metrics": {
            "rmsle": {"reference": "dsio.experimental.model.objectives:RootMeanSquaredError"}
        },
    },
}


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.regression:RegressionOutput",
    "parameters": {
        "shape": [HORIZON_DAYS],
        "inverse": "expm1",
        "non_negative": True,
        "raw_field": "log_prediction",
    },
}
