"""Objectives over native losses: one supervised objective and its auxiliary metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Metric

from dsio.config.components import (
    ComponentError,
    resolve_component,
    validate_component_config,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64}
_STAGES = ("train", "validate", "test")
# Native loss parameters that take a tensor; configuration carries them as numbers.
_TENSOR_PARAMETERS = ("weight", "pos_weight")
# The objective decides how per-element values reduce.
_OWNED_PARAMETERS = ("reduction", "size_average", "reduce")
# Native losses whose integer target is one class index per sample, without the class axis.
_CLASS_INDEX_LOSSES = (nn.CrossEntropyLoss, nn.NLLLoss, nn.MultiMarginLoss)
_RESERVED_NAMES = frozenset({"loss", "loss_step", "loss_epoch"})


class SupervisedObjective(nn.Module):
    """Compute a native loss between the model's prediction and a declared batch field.

    Without sample weights the loss keeps its native reduction, so class weights given as
    ``weight`` behave exactly as in PyTorch (``CrossEntropyLoss`` takes their weighted
    mean). With ``sample_weighted``, the loss is built with ``reduction="none"`` and
    reduced as ``mean(w * loss)``: a class weight then scales its samples' losses with no
    per-batch renormalization, and targets equal to the loss's ``ignore_index`` are
    refused, since they would still count in the mean. Sample weights must already have
    training-role mean 1 and are never renormalized per micro-batch, so ``DsioModule``'s
    logged epoch values (batch-size-weighted means) do not depend on how samples are
    partitioned, and accumulated gradients match the full batch for equal-size
    micro-batches.

    Consumes:
        A batch with ``x`` (the model input), the target field, and, when
        ``sample_weighted``, ``sample_weight`` shaped ``[batch]`` or ``[batch, 1]``.

    Produces:
        ``{"loss": scalar}`` plus one detached scalar per auxiliary metric, which
        ``DsioModule`` logs as ``<stage>/<name>``.

    Parameters:
        ``loss``: component configuration of a native loss module (``torch.nn:MSELoss``,
        ``torch.nn:CrossEntropyLoss``, ...). Its parameters are native; numeric ``weight``
        and ``pos_weight`` become float32 tensors, and ``reduction`` (with the legacy
        ``size_average``/``reduce``) is owned by the objective. ``target``: the batch
        field to predict (default ``y``; ``x`` makes a reconstruction objective).
        ``target_dtype``: optional cast (``float32``, ``float64``, ``int64``).
        ``target_shape``: optional per-sample reshape, e.g. ``[]`` turns a ``[batch, 1]``
        class column into ``[batch]``. ``metrics``: component configurations of stateless
        ``(prediction, target)`` modules, keyed by identifier names. ``metric_stages``: the
        stages that compute metrics (default every stage). ``sample_weighted``: weight each
        sample's loss and metrics (default ``false``).

    Devices:
        CPU and accelerators; class weights are buffers of the loss and move with it.

    Limitations:
        After adaptation, an integer target for a class-index loss (``CrossEntropyLoss``,
        ``NLLLoss``, ``MultiMarginLoss``) must match the prediction without its class
        dimension, and every other target must match the prediction's shape exactly, so a
        target is never silently broadcast. Sample weighting needs a loss and metrics that
        accept ``reduction="none"``. Stateful TorchMetrics are refused as metrics. Masked
        dense targets need a masked objective. Epoch values are logged per process.

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
        # Native class-index losses may divide by class-weight or non-ignored-target sums,
        # not batch size. Only opt into DsioModule's sample-count normalization when the
        # objective owns a mean-over-samples reduction or the native mean is independent
        # of target values.
        self._sample_mean_loss = sample_weighted or not isinstance(self.loss, _CLASS_INDEX_LOSSES)
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
        ignored = getattr(self.loss, "ignore_index", None)
        if (
            weights is not None
            and ignored is not None
            and not target.is_floating_point()
            and bool((target == ignored).any())
        ):
            raise ValueError(
                f"sample-weighted targets cannot use ignore_index {ignored}: "
                "ignored samples would still count in the weighted mean"
            )
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
        if isinstance(self.loss, _CLASS_INDEX_LOSSES) and not target.is_floating_point():
            expected = (prediction.shape[0], *prediction.shape[2:])
        else:
            expected = tuple(prediction.shape)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"target {self.target!r} has shape {tuple(target.shape)}; prediction "
                f"{tuple(prediction.shape)} needs {expected} (declare target_shape to adapt)"
            )
        return target

    def _weights(self, batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
        weights = batch.get("sample_weight")
        if not isinstance(weights, Tensor):
            raise ValueError("a sample-weighted objective needs a sample_weight tensor")
        if weights.ndim not in (1, 2) or (weights.ndim == 2 and weights.shape[1] != 1):
            raise ValueError(
                f"sample_weight must be [batch] or [batch, 1], got {tuple(weights.shape)}"
            )
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
    probe = nn.ModuleDict()
    for name in configs:
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in _RESERVED_NAMES
            or hasattr(probe, name)
        ):
            raise ValueError(
                f"metric names must be identifiers, not {sorted(_RESERVED_NAMES)} or a "
                f"module attribute: {name!r}"
            )
    return configs


def _build(role: str, config: Mapping[str, Any], sample_weighted: bool) -> nn.Module:
    validated = validate_component_config(config)
    parameters = dict(validated["parameters"])
    owned = sorted(set(parameters) & set(_OWNED_PARAMETERS))
    if owned:
        raise ValueError(f"{role}: {owned} owned by the objective")
    runtime: dict[str, Any] = {
        name: torch.atleast_1d(torch.as_tensor(parameters.pop(name), dtype=torch.float32))
        for name in _TENSOR_PARAMETERS
        if isinstance(parameters.get(name), list | int | float)
        and not isinstance(parameters.get(name), bool)
    }
    if sample_weighted:
        runtime["reduction"] = "none"
    try:
        module = resolve_component(
            {"reference": validated["reference"], "parameters": parameters},
            expected=nn.Module,
            **runtime,
        )
    except ComponentError as error:
        hint = " (sample weighting needs reduction='none')" if sample_weighted else ""
        raise ComponentError(f"{role}: {error}{hint}") from error
    if isinstance(module, Metric):
        raise ValueError(f"{role}: stateful TorchMetrics are not supported as metrics")
    return module


def _reduce(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values
    if values.ndim == 0 or values.shape[0] != weights.shape[0]:
        raise ValueError(
            f"a sample-weighted loss must return per-sample values, got {tuple(values.shape)}"
        )
    weights = weights.to(values.dtype).reshape(-1, *([1] * (values.ndim - 1)))
    return (values * weights).mean()


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
    ):
        raise ValueError(f"target_shape must be positive integers, got {shape!r}")
    return tuple(shape)


__all__ = ["RootMeanSquaredError", "SupervisedObjective"]
