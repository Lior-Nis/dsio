"""Configurable model compositions: an MLP block, a backbone/head chain, and stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import resolve_component

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


class MLP(nn.Sequential):
    """Flatten a fixed-shape input, then Linear layers with activations between them.

    Layers are built in order (``Flatten``, then ``Linear`` + activation per hidden width,
    then the output ``Linear`` and an optional output activation), so a seeded MLP
    initializes exactly like the hand-written ``nn.Sequential`` it replaces.

    Consumes:
        A tensor ``[batch, *input_shape]``; integer inputs are cast to float32.

    Produces:
        ``[batch, output]`` after the optional output activation.

    Parameters:
        ``input_shape``: the per-sample shape, checked on every call; ``output``: output
        width; ``hidden``: hidden widths (default none: a single linear layer);
        ``activation``: between layers (``relu``, ``gelu``, ``tanh``, ``sigmoid``,
        ``softplus``; default ``relu``); ``output_activation``: optional, same choices
        (``softplus`` bounds a non-negative regression output).

    Devices:
        CPU and accelerators.

    Limitations:
        Fixed-shape inputs only; variable-length sequences need a sequence encoder.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> model = MLP(input_shape=[1, 4], hidden=[8], output=2, output_activation="softplus")
        >>> tuple(model(torch.ones(3, 1, 4)).shape), bool((model(torch.ones(3, 1, 4)) > 0).all())
        ((3, 2), True)
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        output: int,
        hidden: Sequence[int] = (),
        activation: str = "relu",
        output_activation: str | None = None,
    ) -> None:
        shape = tuple(input_shape)
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
        ):
            raise ValueError("input_shape must be positive integers")
        for name in (activation, output_activation):
            if name is not None:
                _activation(name)  # validate even when no hidden layer would use it
        widths = [_positive("output", output), *(_positive("hidden", width) for width in hidden)]
        layers: list[nn.Module] = [nn.Flatten()]
        features = 1
        for size in shape:
            features *= size
        for width in widths[1:]:
            layers += [nn.Linear(features, width), _activation(activation)]
            features = width
        layers.append(nn.Linear(features, widths[0]))
        if output_activation is not None:
            layers.append(_activation(output_activation))
        super().__init__(*layers)
        self.input_shape = shape

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != len(self.input_shape) + 1 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"MLP expects [batch, {', '.join(map(str, self.input_shape))}], "
                f"got {tuple(x.shape)}"
            )
        return super().forward(x if x.is_floating_point() else x.float())


class Stages(nn.Sequential):
    """Run configured components one after another.

    Each stage is a ``{"reference": "module:qualname", "parameters": {...}}`` mapping, so a
    composition of native modules (``torch.nn:ReLU``, ``torch.nn:Unflatten``) and DSio
    blocks stays plain configuration that provenance records exactly.

    Consumes:
        Whatever the first stage consumes.

    Produces:
        Whatever the last stage produces.

    Parameters:
        ``stages``: a non-empty list of component configurations, built in order.

    Devices:
        CPU and accelerators, as the stages allow.

    Limitations:
        Strictly sequential; branching models need a dedicated composition. An error raised
        inside a stage carries a note naming that stage's position and class.

    Example:
        >>> import torch
        >>> unflatten = {"dim": 1, "unflattened_size": [2, 2]}
        >>> stages = Stages(stages=[
        ...     {"reference": "torch.nn:ReLU"},
        ...     {"reference": "torch.nn:Unflatten", "parameters": unflatten},
        ... ])
        >>> stages(torch.tensor([[-1.0, 2.0, -3.0, 4.0]])).tolist()
        [[[0.0, 2.0], [0.0, 4.0]]]
    """

    def __init__(self, stages: Sequence[Mapping[str, Any]]) -> None:
        if not stages:
            raise ValueError("stages must contain at least one component")
        super().__init__(*(resolve_component(stage, expected=nn.Module) for stage in stages))

    def forward(self, x: Tensor) -> Tensor:
        for index, stage in enumerate(self):
            x = _run(f"Stages stage {index}", stage, x)
        return x


class Chain(nn.Module):
    """Preprocessor, backbone and head as one model that can also ``encode``.

    ``forward`` is ``head(backbone(preprocessor(x)))``; ``encode`` stops before the head,
    which is what self-supervised pretraining exports and a downstream classifier reuses.
    With ``frozen_backbone``, the backbone takes no gradient and runs under
    ``torch.no_grad()``, so only the head trains on top of transferred features. An error
    raised inside a stage carries a note naming the stage (preprocessor, backbone or head).

    Consumes:
        What the preprocessor (or, without one, the backbone) consumes.

    Produces:
        The head's output; ``encode`` returns the backbone's.

    Parameters:
        ``backbone`` and ``head``: component configurations; ``preprocessor``: optional
        configuration (e.g. a ``Standardize`` stage); ``frozen_backbone``: default
        ``false``.

    Devices:
        CPU and accelerators.

    Limitations:
        Loading pretrained backbone weights is the consumer's handoff for now; a verified
        load-and-freeze path is planned with pretrained weights (roadmap v2).

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> model = Chain(
        ...     backbone={"reference": "dsio.experimental.model.compositions:MLP",
        ...               "parameters": {"input_shape": [4], "output": 3}},
        ...     head={"reference": "torch.nn:Linear",
        ...           "parameters": {"in_features": 3, "out_features": 1}},
        ... )
        >>> tuple(model(torch.ones(2, 4)).shape), tuple(model.encode(torch.ones(2, 4)).shape)
        ((2, 1), (2, 3))
    """

    def __init__(
        self,
        backbone: Mapping[str, Any],
        head: Mapping[str, Any],
        preprocessor: Mapping[str, Any] | None = None,
        frozen_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.preprocessor = (
            None if preprocessor is None else resolve_component(preprocessor, expected=nn.Module)
        )
        self.backbone = resolve_component(backbone, expected=nn.Module)
        self.head = resolve_component(head, expected=nn.Module)
        if not isinstance(frozen_backbone, bool):
            raise ValueError("frozen_backbone must be true or false")
        self.frozen_backbone = frozen_backbone
        if frozen_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def encode(self, x: Tensor) -> Tensor:
        if self.preprocessor is not None:
            x = _run("Chain preprocessor", self.preprocessor, x)
        if self.frozen_backbone:
            with torch.no_grad():
                return _run("Chain backbone", self.backbone, x)
        return _run("Chain backbone", self.backbone, x)

    def forward(self, x: Tensor) -> Tensor:
        return _run("Chain head", self.head, self.encode(x))


def _run(stage: str, module: nn.Module, x: Tensor) -> Tensor:
    """Call one stage, naming it on any error so a shape mismatch points at its stage."""
    try:
        return module(x)
    except Exception as error:
        error.add_note(f"in {stage} ({type(module).__name__}), input {tuple(x.shape)}")
        raise


def _activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {name!r}")
    return _ACTIVATIONS[name]()


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} widths must be positive integers, got {value!r}")
    return value


__all__ = ["MLP", "Chain", "Stages"]
