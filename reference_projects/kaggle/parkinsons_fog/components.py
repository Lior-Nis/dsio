"""Dense multi-target components for the Parkinson's consumer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import ComponentConfig

FEATURE_COUNT = 3
TARGET_COUNT = 3
OBSERVED_CHANNEL = FEATURE_COUNT
CHANNEL_COUNT = FEATURE_COUNT + 1 + TARGET_COUNT + 1

DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "columns": [0, FEATURE_COUNT + 1]},
        "y": {
            "from": "data",
            "columns": [FEATURE_COUNT + 1, FEATURE_COUNT + 1 + TARGET_COUNT],
            "finite": True,
            "allowed_values": [0, 1],
        },
        "mask": {
            "from": "data",
            "columns": [CHANNEL_COUNT - 1, CHANNEL_COUNT],
            "dtype": "bool",
            "shape": [-1],
            "finite": True,
            "allowed_values": [0, 1],
        },
    },
}
INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data", "columns": [0, FEATURE_COUNT + 1]}},
}


COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0.0, "y": 0.0, "mask": False},
        "fixed_fields": [],
        "emit_mask": False,
    },
}
INPUT_COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0.0},
        "fixed_fields": [],
        "emit_mask": False,
    },
}
OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.masked_objective:MaskedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:BCEWithLogitsLoss"},
        "target_dtype": "float32",
    },
}
MODEL: ComponentConfig = {
    "reference": "dsio.experimental.model.compositions:Chain",
    "parameters": {
        "preprocessor": {
            "reference": "dsio.experimental.model.compositions:Stages",
            "parameters": {
                "stages": [
                    {
                        "reference": "dsio.experimental.model.layout:TimeMajorToChannelFirst",
                        "parameters": {"channels": FEATURE_COUNT + 1},
                    },
                    {
                        "reference": (
                            "dsio.experimental.model.standardization:InstanceStandardize"
                        ),
                        "parameters": {"observed_channel": OBSERVED_CHANNEL},
                    },
                ]
            },
        },
        "backbone": {
            "reference": "dsio.experimental.model.convolution:DenseConv1d",
            "parameters": {
                "channels": FEATURE_COUNT,
                "hidden": 16,
                "output": TARGET_COUNT,
                "kernel_size": 5,
                "observed_channel": OBSERVED_CHANNEL,
            },
        },
        "head": {
            "reference": "dsio.experimental.model.layout:ChannelFirstToTimeMajor",
            "parameters": {"channels": TARGET_COUNT},
        },
    },
}


class FogOutput(nn.Module):
    def forward(self, logits: Tensor) -> Mapping[str, Tensor]:
        probability = torch.sigmoid(logits)
        return {"prediction": (probability >= 0.5).long(), "probability": probability}


def validate_fog_prediction(output: Mapping[str, Any]) -> None:
    prediction = output.get("prediction")
    probability = output.get("probability")
    if not isinstance(prediction, Tensor) or prediction.ndim != 3:
        raise ValueError("FoG prediction must have shape [batch, points, classes]")
    if not isinstance(probability, Tensor) or probability.ndim != 3:
        raise ValueError("FoG probability must have shape [batch, points, classes]")
    if prediction.shape != probability.shape or probability.shape[2] != TARGET_COUNT:
        raise ValueError(f"FoG probability must have {TARGET_COUNT} classes")
    if not bool(((prediction == 0) | (prediction == 1)).all()):
        raise ValueError("FoG prediction must be binary")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("FoG probability must be finite")
    if not bool(((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("FoG probability must be between zero and one")
    if not torch.equal(prediction, (probability >= 0.5).long()):
        raise ValueError("FoG prediction must equal probability thresholded at 0.5")
