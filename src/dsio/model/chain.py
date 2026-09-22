"""Reusable components for the established encoder/head training pattern."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class ComponentError(ValueError):
    """An encoder/head component chain is incomplete or incompatible."""


class ComponentChain(nn.Module):
    """Compose optional preprocessing with a transform, backbone, and task head."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module,
        transform: nn.Module | None = None,
        preprocessor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        for name, component in (("backbone", backbone), ("head", head)):
            if not isinstance(component, nn.Module):
                raise ComponentError(f"{name} is required and must be a torch nn.Module")
        for name, optional in (("transform", transform), ("preprocessor", preprocessor)):
            if optional is not None and not isinstance(optional, nn.Module):
                raise ComponentError(f"{name} must be a torch nn.Module or None")
        self.preprocessor = preprocessor
        self.transform = transform if transform is not None else nn.Identity()
        self.backbone = backbone
        self.head = head

    def encode(self, x: Tensor) -> Tensor:
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        return self.backbone(self.transform(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


class LossObjective(nn.Module):
    """Adapt an established `(prediction, target)` loss to the generic objective contract."""

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        if not isinstance(loss, nn.Module):
            raise ComponentError("loss must be a torch nn.Module")
        self.loss = loss

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        stage: str,
    ) -> Mapping[str, Tensor]:
        del stage
        try:
            x = batch["x"]
            target = batch["y"]
        except KeyError as error:
            raise ComponentError(f"loss objective batch is missing {error.args[0]!r}") from None
        prediction = model(x)
        value = self.loss(prediction, target)
        if not isinstance(value, Tensor):
            raise ComponentError(f"loss must return a tensor, got {type(value).__name__}")
        if value.ndim > 0:
            value = value.mean()
        result: dict[str, Tensor] = {"loss": value}
        diagnostics = getattr(self.loss, "diagnostics", None)
        if callable(diagnostics):
            reported = diagnostics(prediction, target, x)
            if not isinstance(reported, Mapping):
                raise ComponentError(
                    f"loss diagnostics must return a mapping, got {type(reported).__name__}"
                )
            if "loss" in reported:
                raise ComponentError("loss diagnostics cannot replace the optimization loss")
            result.update(reported)
        return result


def export_encoder(module: Any) -> dict[str, Tensor]:
    """Export the encoder portion of a DSio module configured with `ComponentChain`."""
    model = getattr(module, "model", None)
    if not isinstance(model, ComponentChain):
        raise ComponentError("encoder export requires a ComponentChain model")
    state: dict[str, torch.Tensor] = {}
    for name in ("preprocessor", "transform", "backbone"):
        component = getattr(model, name, None)
        if isinstance(component, nn.Module):
            for key, value in component.state_dict().items():
                state[f"{name}.{key}"] = value
    return state
