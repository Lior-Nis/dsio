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

#: Stochastic view-builders for a contrastive SSL objective (SimCLR, VICReg), which need
#: two independently-augmented views of the same window to pull together or push apart.
#: Not wired into :class:`~dsio.nn.module.DsioModule` — the chain has no stochastic slot,
#: so nothing here can leak into a validation batch by accident. A pretext objective that
#: instead needs one masked view (MAE) gets it from the dataset, not from here.
VIEW_AUGMENTORS: Registry[ComponentFactory] = Registry("view_augmentor")

#: Fitted-on-train preprocessing, kept separate from transforms because it has state.
PREPROCESSORS: Registry[ComponentFactory] = Registry("preprocessor")

#: Per-row label providers over a store, keyed by name.
#: Unlike the registries above, this one intentionally ships with no built-in entries —
#: projects supply their own labels, and ``check_torch`` fails loudly via ``LABELS.get``
#: when one is missing, so an empty ``LABELS`` is not the same defect as an empty
#: ``BACKBONES``/``HEADS``/``LOSSES`` (see ``tests/nn/test_registry_bootstrap.py``).
LABELS: Registry[Callable[..., Any]] = Registry("labels")


def backbone(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return BACKBONES.register(name)


def head(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return HEADS.register(name)


def loss(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return LOSSES.register(name)


def transform(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return TRANSFORMS.register(name)


def view_augmentor(name: str) -> Callable[[ComponentFactory], ComponentFactory]:
    return VIEW_AUGMENTORS.register(name)


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
