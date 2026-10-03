"""Importable Digit Recognizer representation and classifier components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import ComponentConfig

UNLABELLED_DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32", "divide": 255.0},
    },
}

LABELLED_DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32", "divide": 255.0},
        "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
    },
}


# The encoder pretrains inside the autoencoder and transfers, frozen, into the classifier.
ENCODER: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:MLP",
    "parameters": {"input_shape": [28, 28], "hidden": [32], "output": 16},
}

AUTOENCODER: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:Chain",
    "parameters": {
        "backbone": ENCODER,
        "head": {
            "reference": "dsio.experimental.model.compositions:Stages",
            "parameters": {
                "stages": [
                    {"reference": "torch.nn:ReLU"},
                    {
                        "reference": "dsio.experimental.model.compositions:MLP",
                        "parameters": {
                            "input_shape": [16],
                            "hidden": [32],
                            "output": 784,
                            "output_activation": "sigmoid",
                        },
                    },
                    {
                        "reference": "torch.nn:Unflatten",
                        "parameters": {"dim": 1, "unflattened_size": [28, 28]},
                    },
                ]
            },
        },
    },
}

CLASSIFIER: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:Chain",
    "parameters": {
        "backbone": ENCODER,
        "head": {
            "reference": "torch.nn:Linear",
            "parameters": {"in_features": 16, "out_features": 10},
        },
        "frozen_backbone": True,
    },
}


# Reconstruction: the input is the target.
RECONSTRUCTION_OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {"loss": {"reference": "torch.nn:MSELoss"}, "target": "x"},
}


CLASSIFICATION_OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {"loss": {"reference": "torch.nn:CrossEntropyLoss"}},
}


class DigitPrediction(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        if logits.ndim != 2 or logits.shape[1] != 10 or not bool(torch.isfinite(logits).all()):
            raise ValueError("digit logits must be a finite [batch, 10] tensor")
        return {"prediction": torch.argmax(logits, dim=1)}


def validate_digit_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    if (
        not isinstance(prediction, Tensor)
        or prediction.ndim != 1
        or prediction.dtype != torch.int64
    ):
        raise ValueError("digit prediction must be an int64 [batch] tensor")
    if not bool(torch.all((prediction >= 0) & (prediction <= 9))):
        raise ValueError("digit predictions must be in [0, 9]")
