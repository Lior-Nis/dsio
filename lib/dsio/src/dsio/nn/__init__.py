"""Torch components, assembled into one chain and trained by one step.

Importing this package registers dsio's built-in components. torch and lightning are an
optional extra, so nothing outside :mod:`dsio.nn` and the torch runner imports it.
"""

from dsio.nn.components import (
    Conv1dEncoder,
    CrossEntropy,
    FixedStandardize,
    InstanceStandardize,
    Jitter,
    MLP1d,
    RandomScale,
)
from dsio.nn.data import WindowDataset, make_loader
from dsio.nn.module import ComponentError, DsioModule, Stage
from dsio.nn.registry import (
    AUGMENTORS,
    BACKBONES,
    HEADS,
    LABELS,
    LOSSES,
    PREPROCESSORS,
    TRANSFORMS,
    augmentor,
    backbone,
    head,
    labels,
    loss,
    preprocessor,
    transform,
)

__all__ = [
    "AUGMENTORS",
    "BACKBONES",
    "HEADS",
    "LABELS",
    "LOSSES",
    "PREPROCESSORS",
    "TRANSFORMS",
    "ComponentError",
    "Conv1dEncoder",
    "CrossEntropy",
    "DsioModule",
    "FixedStandardize",
    "InstanceStandardize",
    "Jitter",
    "MLP1d",
    "RandomScale",
    "Stage",
    "WindowDataset",
    "augmentor",
    "backbone",
    "head",
    "labels",
    "loss",
    "make_loader",
    "preprocessor",
    "transform",
]
