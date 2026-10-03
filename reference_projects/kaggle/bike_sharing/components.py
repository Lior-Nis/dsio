"""Importable Bike Sharing training and inference components."""

from __future__ import annotations

from collections.abc import Sequence

from dsio.config.components import ComponentConfig
from dsio.experimental.model import Stages

FEATURES = 9


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


def standardized_mlp(*, features: int, mean: Sequence[float], scale: Sequence[float]) -> Stages:
    """Construct the configured architecture with explicitly injected fitted statistics."""
    if features != FEATURES:
        raise ValueError(f"Bike Sharing model requires {FEATURES} features")
    return Stages(
        stages=[
            {
                "reference": "dsio.experimental.model.standardization:Standardize",
                "parameters": {"mean": list(mean), "scale": list(scale)},
            },
            {
                "reference": "dsio.experimental.model.compositions:MLP",
                "parameters": {
                    "input_shape": [1, features],
                    "output": 1,
                    "output_activation": "softplus",
                },
            },
        ]
    )


MODEL: ComponentConfig = {
    "reference": "reference_projects.kaggle.bike_sharing.components:standardized_mlp",
    "parameters": {"features": FEATURES},
}


OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "metrics": {"mae": {"reference": "torch.nn:L1Loss"}},
    },
}


OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.regression:RegressionOutput",
    "parameters": {"shape": [1], "non_negative": True},
}
