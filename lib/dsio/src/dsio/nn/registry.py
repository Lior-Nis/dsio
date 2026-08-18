"""Registries for the pieces a torch model is assembled from.

One registry per slot in the component chain, rather than one registry of "models". A
single model registry forces every new objective to touch every place that knows how models
are built. When each slot is independently registered, a new backbone is one decorator and a
new loss is one decorator, and neither knows the other exists.

Every factory takes keyword arguments only and returns an ``nn.Module``. Keyword-only is
deliberate — positional arguments in a config file are unreadable six months later, and
they make a factory's signature part of its public contract in a way that renaming a
parameter silently breaks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dsio.config.registry import Registry

ComponentFactory = Callable[..., Any]

#: Feature extractors. Take ``[batch, channels, time]`` and return features.
BACKBONES: Registry[ComponentFactory] = Registry("backbone")

#: Task-specific output layers. Always present, never ``None``.
HEADS: Registry[ComponentFactory] = Registry("head")

#: Objectives. Take ``(prediction, target)`` and return a per-sample loss.
LOSSES: Registry[ComponentFactory] = Registry("loss")

#: Deterministic signal transforms — resampling, spectrograms, normalisation.
TRANSFORMS: Registry[ComponentFactory] = Registry("transform")

#: Stochastic augmentations. Applied in training only; see :mod:`dsio.nn.module`.
AUGMENTORS: Registry[ComponentFactory] = Registry("augmentor")

#: Fitted-on-train preprocessing, kept separate from transforms because it has state.
PREPROCESSORS: Registry[ComponentFactory] = Registry("preprocessor")

#: Per-row label providers over a store, keyed by name.
LABELS: Registry[Callable[..., Any]] = Registry("labels")


def backbone(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return BACKBONES.register(name)


def head(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return HEADS.register(name)


def loss(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return LOSSES.register(name)


def transform(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return TRANSFORMS.register(name)


def augmentor(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return AUGMENTORS.register(name)


def preprocessor(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return PREPROCESSORS.register(name)


def labels(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a per-row label provider.

    Labels live outside the store rather than inside it. The store is a canonical record of
    what was *measured*; a label is an interpretation of it, and interpretations get revised
    — a relabelled cohort must not force a re-ingest of the signal, and two labelling
    schemes over one corpus must not mean two copies of the bytes.
    """
    return LABELS.register(name)
