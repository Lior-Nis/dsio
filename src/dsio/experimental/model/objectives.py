"""Objectives over native losses: one supervised objective and its auxiliary metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dsio.config.components import (
    ComponentError,
    resolve_component,
    validate_component_config,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64}
_STAGES = ("train", "validate", "test")
# Native loss parameters that take a tensor; configuration carries them as number lists.
_TENSOR_PARAMETERS = ("weight", "pos_weight")


class SupervisedObjective(nn.Module):
    """Compute a native loss between the model's prediction and a declared batch field.

    The loss keeps its native semantics: class weights given as ``weight`` behave exactly
    as in PyTorch (``CrossEntropyLoss`` takes their weighted mean). With
    ``sample_weighted``, the loss is built with ``reduction="none"`` and reduced as
    ``mean(w * loss)``. Sample weights must already have training-role mean 1; they are
    never renormalized per micro-batch, so the loss does not depend on how samples are
    partitioned into batches.

    Consumes:
        A batch with ``x`` (the model input), the target field, and, when
        ``sample_weighted``, ``sample_weight`` shaped ``[batch]`` or ``[batch, 1]``.

    Produces:
        ``{"loss": scalar}`` plus one detached scalar per auxiliary metric, which
        ``DsioModule`` logs as ``<stage>/<name>``.

    Parameters:
        ``loss``: component configuration of a native loss module (``torch.nn:MSELoss``,
        ``torch.nn:CrossEntropyLoss``, ...). Its parameters are native; list-valued
        ``weight`` and ``pos_weight`` become float32 tensors, and ``reduction`` is owned
        by the objective. ``target``: the batch field to predict (default ``y``; ``x``
        makes a reconstruction objective). ``target_dtype``: optional cast (``float32``,
        ``float64``, ``int64``). ``target_shape``: optional per-sample reshape, e.g.
        ``[]`` turns a ``[batch, 1]`` class column into ``[batch]``. ``metrics``: named
        component configurations of ``(prediction, target)`` modules. ``metric_stages``:
        the stages that compute metrics (default every stage). ``sample_weighted``:
        weight each sample's loss and metrics (default ``false``).

    Devices:
        CPU and accelerators; class weights are buffers of the loss and move with it.

    Limitations:
        After adaptation, a floating target must match the prediction's shape exactly and
        an integer target must match it without the class dimension, so a ``[batch, 1]``
        target is never silently broadcast against a ``[batch]`` prediction. Sample
        weighting needs a loss and metrics that accept ``reduction="none"``. Masked dense
        targets need a masked objective.

    Example:
        >>> import torch
        >>> objective = SupervisedObjective(
        ...     loss={"reference": "torch.nn:MSELoss"},
        ...     metrics={"mae": {"reference": "torch.nn:L1Loss"}},
        ... )
        >>> batch = {"x": torch.tensor([[1.0], [3.0]]), "y": torch.tensor([[2.0], [3.0]])}
        >>> result = objective(torch.nn.Identity(), batch, "train")
        >>> {name: value.item() for name, value in result.items()}
        {'loss': 0.5, 'mae': 0.5}
    """

    def __init__(
        self,
        loss: Mapping[str, Any],
        target: str = "y",
        target_dtype: str | None = None,
        target_shape: Sequence[int] | None = None,
        metrics: Mapping[str, Mapping[str, Any]] | None = None,
        metric_stages: Sequence[str] | None = None,
        sample_weighted: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(target, str) or not target:
            raise ValueError("target must name a batch field")
        if target_dtype is not None and target_dtype not in _DTYPES:
            raise ValueError(f"target_dtype must be one of {sorted(_DTYPES)}, got {target_dtype!r}")
        if not isinstance(sample_weighted, bool):
            raise ValueError("sample_weighted must be true or false")
        stages = tuple(_STAGES if metric_stages is None else metric_stages)
        unknown = sorted(set(stages) - set(_STAGES))
        if unknown:
            raise ValueError(f"metric_stages must be among {list(_STAGES)}, got {unknown}")
        self.target = target
        self.target_dtype = None if target_dtype is None else _DTYPES[target_dtype]
        self.target_shape = None if target_shape is None else _shape(target_shape)
        self.metric_stages = frozenset(stages)
        self.sample_weighted = sample_weighted
        self.loss = _build("loss", loss, sample_weighted)
        self.metrics = nn.ModuleDict(
            {
                name: _build(f"metric {name!r}", config, sample_weighted)
                for name, config in _metric_configs(metrics).items()
            }
        )

    def forward(self, model: nn.Module, batch: Mapping[str, Any], stage: str) -> dict[str, Tensor]:
        prediction = model(batch["x"])
        if not isinstance(prediction, Tensor):
            raise ValueError(
                f"the model must return a prediction tensor, got {type(prediction).__name__}"
            )
        target = self._target(batch, prediction)
        weights = self._weights(batch, prediction) if self.sample_weighted else None
        result = {"loss": _reduce(self.loss(prediction, target), weights)}
        if stage in self.metric_stages:
            detached = prediction.detach()
            for name, metric in self.metrics.items():
                result[name] = _reduce(metric(detached, target), weights).detach()
        return result

    def _target(self, batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
        try:
            target = batch[self.target]
        except KeyError:
            raise ValueError(f"objective target field {self.target!r} is missing") from None
        if not isinstance(target, Tensor):
            raise ValueError(f"objective target {self.target!r} must be a tensor")
        if self.target_shape is not None:
            target = target.reshape(target.shape[0], *self.target_shape)
        if self.target_dtype is not None:
            target = target.to(self.target_dtype)
        if target.is_floating_point():
            expected = tuple(prediction.shape)
        else:
            expected = (prediction.shape[0], *prediction.shape[2:])
        if tuple(target.shape) != expected:
            kind = "floating" if target.is_floating_point() else "class-index"
            raise ValueError(
                f"{kind} target {self.target!r} has shape {tuple(target.shape)}; prediction "
                f"{tuple(prediction.shape)} needs {expected} (declare target_shape to adapt)"
            )
        return target

    def _weights(self, batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
        weights = batch.get("sample_weight")
        if not isinstance(weights, Tensor):
            raise ValueError("a sample-weighted objective needs a sample_weight tensor")
        weights = weights.reshape(-1)
        if weights.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"sample_weight has {weights.shape[0]} values for {prediction.shape[0]} samples"
            )
        if not weights.is_floating_point():
            weights = weights.float()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("sample weights must be finite and non-negative")
        return weights


class RootMeanSquaredError(nn.Module):
    """Square root of the mean squared error, as an auxiliary metric.

    Consumes:
        ``(prediction, target)`` tensors of the same shape.

    Produces:
        A scalar: ``sqrt(mse(prediction, target))``; on log1p targets this is RMSLE.

    Parameters:
        None.

    Devices:
        CPU and accelerators.

    Limitations:
        Not elementwise, so it cannot be sample-weighted; a micro-batch value is the root
        of that batch's mean, and epoch logging averages those roots.

    Example:
        >>> import torch
        >>> RootMeanSquaredError()(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0])).item()
        1.4142135381698608
    """

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(F.mse_loss(prediction, target))


def _metric_configs(metrics: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    configs = dict(metrics or {})
    for name in configs:
        if not isinstance(name, str) or not name or "/" in name or name == "loss":
            raise ValueError(f"metric names must be non-empty, without '/', not 'loss': {name!r}")
    return configs


def _build(role: str, config: Mapping[str, Any], sample_weighted: bool) -> nn.Module:
    validated = validate_component_config(config)
    parameters = dict(validated["parameters"])
    if "reduction" in parameters:
        raise ValueError(f"{role}: reduction is owned by the objective")
    runtime: dict[str, Any] = {
        name: torch.as_tensor(parameters.pop(name), dtype=torch.float32)
        for name in _TENSOR_PARAMETERS
        if isinstance(parameters.get(name), list)
    }
    if sample_weighted:
        runtime["reduction"] = "none"
    try:
        return resolve_component(
            {"reference": validated["reference"], "parameters": parameters},
            expected=nn.Module,
            **runtime,
        )
    except ComponentError as error:
        hint = " (sample weighting needs reduction='none')" if sample_weighted else ""
        raise ComponentError(f"{role}: {error}{hint}") from error


def _reduce(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values
    if values.ndim == 0 or values.shape[0] != weights.shape[0]:
        raise ValueError(
            f"a sample-weighted loss must return per-sample values, got {tuple(values.shape)}"
        )
    return (values * weights.reshape(-1, *([1] * (values.ndim - 1)))).mean()


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
    ):
        raise ValueError(f"target_shape must be positive integers, got {shape!r}")
    return tuple(shape)


__all__ = ["RootMeanSquaredError", "SupervisedObjective"]
