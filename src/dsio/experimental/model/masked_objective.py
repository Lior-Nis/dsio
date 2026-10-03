"""A dense objective that selects valid positions before invoking native losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from dsio.experimental.model import objectives


class MaskedObjective(nn.Module):
    """Compute a native loss and metrics only over True positions of ``batch['mask']``.

    Prediction and target values are selected before the configured modules run, so an
    ignored non-finite target cannot contaminate the loss or its gradients. A mask may
    match the prediction exactly or omit its final output axis; no other broadcasting is
    inferred.

    Consumes:
        A batch with model input ``x``, target ``y``, and boolean True-is-valid ``mask``.
        Prediction and target shapes must be identical. The mask is either that same shape
        or the shape without its final output axis.

    Produces:
        ``{"loss": scalar}`` plus detached configured scalar metrics for the current stage.

    Parameters:
        ``loss``: component configuration of a native loss module. Native parameters are
        supported except reduction parameters, which the objective owns. ``target_dtype``:
        optional target cast (``float32``, ``float64``, or ``int64``). ``metrics``: named
        component configurations over selected prediction/target values. ``metric_stages``:
        stages that compute metrics (default train, validate, and test).

    Devices:
        CPU and accelerators; prediction, target, and mask must share a device.

    Limitations:
        The reduction is a mean over selected values, not samples, so sample-normalized
        gradient accumulation is unsupported. Stateful TorchMetrics are refused. A
        full-shape mask cannot be combined with multi-value native ``weight`` or
        ``pos_weight`` tensors because selection would flatten their output axis. The
        mask cannot omit more than the final output axis.

    Example:
        >>> import torch
        >>> objective = MaskedObjective(loss={"reference": "torch.nn:MSELoss"})
        >>> batch = {
        ...     "x": torch.tensor([[1.0, 99.0], [3.0, 99.0]]),
        ...     "y": torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
        ...     "mask": torch.tensor([[True, False], [True, False]]),
        ... }
        >>> objective(torch.nn.Identity(), batch, "train")["loss"].item()
        2.5
    """

    def __init__(
        self,
        loss: Mapping[str, Any],
        target_dtype: str | None = None,
        metrics: Mapping[str, Mapping[str, Any]] | None = None,
        metric_stages: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if target_dtype is not None and target_dtype not in objectives._DTYPES:
            raise ValueError(
                f"target_dtype must be one of {sorted(objectives._DTYPES)}, got {target_dtype!r}"
            )
        stages = tuple(objectives._STAGES if metric_stages is None else metric_stages)
        unknown = sorted(set(stages) - set(objectives._STAGES))
        if unknown:
            raise ValueError(
                f"metric_stages must be among {list(objectives._STAGES)}, got {unknown}"
            )
        self.target_dtype = None if target_dtype is None else objectives._DTYPES[target_dtype]
        self.metric_stages = frozenset(stages)
        self.loss = objectives._build("loss", loss, sample_weighted=False, reduction="mean")
        self.metrics = nn.ModuleDict(
            {
                name: objectives._build(f"metric {name!r}", config, sample_weighted=False)
                for name, config in objectives._metric_configs(metrics).items()
            }
        )
        self._sample_mean_loss = False

    def forward(self, model: nn.Module, batch: Mapping[str, Any], stage: str) -> dict[str, Tensor]:
        if stage not in objectives._STAGES:
            raise ValueError(f"stage must be one of {list(objectives._STAGES)}, got {stage!r}")
        try:
            model_input = batch["x"]
        except KeyError:
            raise ValueError("objective input field 'x' is missing") from None
        prediction = model(model_input)
        if not isinstance(prediction, Tensor):
            raise ValueError(
                f"the model must return a prediction tensor, got {type(prediction).__name__}"
            )
        target = self._target(batch, prediction)
        mask = self._mask(batch, prediction)
        if tuple(mask.shape) == tuple(prediction.shape):
            self._reject_flattened_weights(self.loss, "loss")
            if stage in self.metric_stages:
                for name, metric in self.metrics.items():
                    self._reject_flattened_weights(metric, f"metric {name!r}")
        selected_prediction = prediction[mask]
        selected_target = target[mask]
        if selected_prediction.numel() == 0:
            raise ValueError("masked objective selected no values")
        result = {"loss": self._scalar(self.loss(selected_prediction, selected_target), "loss")}
        if stage in self.metric_stages:
            detached = selected_prediction.detach()
            for name, metric in self.metrics.items():
                value = self._scalar(metric(detached, selected_target), f"metric {name!r}")
                result[name] = value.detach()
        return result

    @staticmethod
    def _scalar(value: Any, role: str) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 0:
            kind = (
                type(value).__name__ if not isinstance(value, Tensor) else str(tuple(value.shape))
            )
            raise ValueError(f"{role} must return a scalar tensor, got {kind}")
        return value

    @staticmethod
    def _reject_flattened_weights(module: nn.Module, role: str) -> None:
        for name in objectives._TENSOR_PARAMETERS:
            value = getattr(module, name, None)
            if isinstance(value, Tensor) and value.numel() > 1:
                raise ValueError(
                    f"full-shape mask flattens the output axis and cannot preserve "
                    f"{role} {name} with {value.numel()} values; use a mask that omits "
                    "the final output axis"
                )

    def _target(self, batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
        try:
            target = batch["y"]
        except KeyError:
            raise ValueError("objective target field 'y' is missing") from None
        if not isinstance(target, Tensor):
            raise ValueError("objective target 'y' must be a tensor")
        if self.target_dtype is not None:
            target = target.to(self.target_dtype)
        if tuple(target.shape) != tuple(prediction.shape):
            raise ValueError(
                f"target 'y' has shape {tuple(target.shape)}; prediction "
                f"{tuple(prediction.shape)} must match exactly"
            )
        if target.device != prediction.device:
            raise ValueError(
                f"target 'y' is on {target.device}; prediction is on {prediction.device}"
            )
        return target

    @staticmethod
    def _mask(batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
        try:
            mask = batch["mask"]
        except KeyError:
            raise ValueError("objective batch field 'mask' is missing") from None
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise ValueError("objective mask must be a boolean tensor")
        expected = {tuple(prediction.shape)}
        if prediction.ndim > 0:
            expected.add(tuple(prediction.shape[:-1]))
        if tuple(mask.shape) not in expected:
            choices = " or ".join(str(shape) for shape in sorted(expected, key=len))
            raise ValueError(
                f"objective mask shape {tuple(mask.shape)} must match prediction shape {choices}"
            )
        if mask.device != prediction.device:
            raise ValueError(
                f"objective mask is on {mask.device}; prediction is on {prediction.device}"
            )
        if not bool(mask.any()):
            raise ValueError("masked objective has no valid positions")
        return mask


__all__ = ["MaskedObjective"]
