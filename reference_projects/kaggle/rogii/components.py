"""Variable-length dense-regression components for the ROGII consumer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dsio.config.components import ComponentConfig

FEATURES = 13
TARGET_SCALE = 20_000.0


def well_arrays(well: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    tail = well["tail"]
    origin = tail[0]
    type_tvt = np.asarray(well["typewell"]["TVT"], dtype=np.float64)
    type_gr = np.asarray(well["typewell"]["GR"], dtype=np.float64)
    rows = []
    targets = []
    for item in tail:
        gr = item["GR"]
        rows.append(
            [
                (item["MD"] - origin["MD"]) / 10_000,
                (item["X"] - origin["X"]) / 10_000,
                (item["Y"] - origin["Y"]) / 10_000,
                (item["Z"] - origin["Z"]) / 10_000,
                0.0 if gr is None else gr / 200,
                float(gr is None),
                well["last_known_tvt"] / TARGET_SCALE,
                well["last_tvt_slope"],
                float(type_tvt.min()) / TARGET_SCALE,
                float(type_tvt.max()) / TARGET_SCALE,
                float(type_gr.mean()) / 200,
                float(type_gr.std()) / 200,
                1.0,
            ]
        )
        targets.append(0.0 if "TVT" not in item else item["TVT"] / TARGET_SCALE)
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "columns": [0, FEATURES]},
        "y": {"from": "data", "columns": [FEATURES, FEATURES + 1], "shape": [-1]},
    },
}
COLLATOR: ComponentConfig = {
    "reference": "dsio.experimental.data.padding:PadCollator",
    "parameters": {
        "padded_fields": {"x": 0.0, "y": 0.0},
        "fixed_fields": [],
        "emit_mask": True,
    },
}
OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.masked_objective:MaskedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:MSELoss"},
        "target_dtype": "float32",
        "metrics": {
            "rmse": {
                "reference": "dsio.experimental.model.objectives:RootMeanSquaredError",
                "parameters": {"scale": TARGET_SCALE},
            }
        },
    },
}
MODEL: ComponentConfig = {
    "reference": "dsio.experimental.model.residual:BaselineResidual",
    "parameters": {
        "features": FEATURES,
        "hidden": 32,
        "baseline_channel": 6,
        "validity_channel": FEATURES - 1,
        "bound": 0.01,
    },
}
OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.regression:RegressionOutput",
    "parameters": {"shape": [None], "scale": TARGET_SCALE},
}
