"""Standardization slots: fixed statistics applied as a model or predictor stage."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Standardize(nn.Module):
    """Subtract a fixed per-feature mean and divide by a fixed per-feature scale.

    The statistics come from outside, typically
    :func:`~dsio.experimental.data.fitting.fit_standardization` on the training role, or
    are constants (``mean=[0]``, ``scale=[255]`` for 8-bit pixels). Inputs of another dtype
    (integers, float64) are cast to the statistics' dtype first, so the same stage serves
    raw predictor inputs and float training batches.

    Consumes:
        A tensor whose ``axis`` dimension holds the features; any other dimensions
        broadcast. A single-value ``mean``/``scale`` applies to every feature.

    Produces:
        A tensor of the input's shape and the statistics' dtype (float32 unless the module
        is cast): ``(x - mean) / scale``.

    Parameters:
        ``mean`` and ``scale``: equal-length sequences of finite numbers (``scale`` must be
        non-zero); ``axis``: the feature axis (default ``-1``, the last).

    Devices:
        CPU and accelerators; the statistics are buffers that move with the module.

    Limitations:
        Exact division, no epsilon: fitters map zero-variance features to scale 1.
        Missing values (NaN) propagate.

    Example:
        >>> import torch
        >>> stage = Standardize(mean=[1.0, 10.0], scale=[2.0, 5.0])
        >>> stage(torch.tensor([[[3.0, 20.0]]])).tolist()
        [[[1.0, 2.0]]]
        >>> pixels = torch.tensor([[255]], dtype=torch.uint8)
        >>> Standardize(mean=[0.0], scale=[255.0])(pixels).tolist()
        [[1.0]]
    """

    mean: Tensor
    scale: Tensor

    def __init__(self, mean: Sequence[float], scale: Sequence[float], axis: int = -1) -> None:
        super().__init__()
        if len(mean) == 0 or len(mean) != len(scale):
            raise ValueError("mean and scale must be non-empty and the same length")
        if not all(math.isfinite(value) for value in [*mean, *scale]):
            raise ValueError("mean and scale must be finite")
        if any(value == 0 for value in scale):
            raise ValueError("scale must be non-zero; map zero-variance features to 1")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError("axis must be an integer")
        self.axis = axis
        self.register_buffer("mean", torch.tensor(list(mean), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(list(scale), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        values = x if x.dtype == self.mean.dtype else x.to(self.mean.dtype)
        features = self.mean.numel()
        if not -values.ndim <= self.axis < values.ndim:
            raise ValueError(
                f"axis {self.axis} is out of range for input of shape {tuple(x.shape)}"
            )
        if features != 1 and values.shape[self.axis] != features:
            raise ValueError(
                f"expected {features} features on axis {self.axis}, got shape {tuple(x.shape)}"
            )
        shape = [1] * values.ndim
        shape[self.axis] = -1
        return (values - self.mean.view(shape)) / self.scale.view(shape)


class InstanceStandardize(nn.Module):
    """Standardize each sample and channel over its observed timesteps.

    Consumes:
        A floating tensor ``[batch, channels, time]``. When ``observed_channel`` is set,
        that channel must contain zero-or-one validity; invalid value positions may be
        non-finite because they are selected out before arithmetic.

    Produces:
        ``[batch, channels, time]`` with per-instance, per-value-channel population mean
        zero and scale one where variance is non-zero. Invalid positions are exact zero;
        a declared validity channel is preserved for padding-aware downstream blocks.

    Parameters:
        ``eps``: finite positive minimum scale (default ``1e-6``).
        ``observed_channel``: optional channel-axis index containing shared timestep
        validity; negative indexes are accepted. With no channel, every timestep is used.

    Devices:
        CPU and accelerators. Floating dtype/device are preserved; integer inputs become
        float32.

    Limitations:
        Rank-three channel-first signals only. One validity channel is shared by every
        value channel; feature-specific missingness needs a different declared block. Half
        and bfloat16 statistics are accumulated in float32, then cast back.

    Example:
        >>> import torch
        >>> x = torch.tensor([[[1., 3., 99.], [1., 1., 0.]]])
        >>> InstanceStandardize(observed_channel=1)(x).tolist()
        [[[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]
    """

    def __init__(self, eps: float = 1e-6, observed_channel: int | None = None) -> None:
        super().__init__()
        if isinstance(eps, bool) or not isinstance(eps, int | float):
            raise ValueError("eps must be finite and positive")
        try:
            normalized_eps = float(eps)
        except OverflowError:
            raise ValueError("eps must be finite and positive") from None
        if not math.isfinite(normalized_eps) or normalized_eps <= 0:
            raise ValueError("eps must be finite and positive")
        if observed_channel is not None and (
            isinstance(observed_channel, bool) or not isinstance(observed_channel, int)
        ):
            raise ValueError("observed_channel must be an integer or None")
        self.eps = normalized_eps
        self.observed_channel = observed_channel

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"InstanceStandardize expects rank 3 [batch, channels, time], "
                f"got shape {tuple(x.shape)}"
            )
        values = x if x.is_floating_point() else x.float()
        if self.observed_channel is None:
            signal = values
            observed = torch.ones(
                (values.shape[0], 1, values.shape[2]),
                dtype=torch.bool,
                device=values.device,
            )
        else:
            channels = values.shape[1]
            if not -channels <= self.observed_channel < channels or channels < 2:
                raise ValueError(
                    f"observed_channel {self.observed_channel} must identify one of at "
                    f"least two channels, got shape {tuple(values.shape)}"
                )
            index = self.observed_channel % channels
            validity = values[:, index : index + 1]
            torch._check_tensor_all(
                torch.isfinite(validity) & ((validity == 0) | (validity == 1)),
                lambda: "observed validity channel must contain only zero or one",
            )
            observed = validity.bool()
            signal = torch.cat((values[:, :index], values[:, index + 1 :]), dim=1)
        count = observed.sum(dim=2, keepdim=True)
        torch._check_tensor_all(
            count > 0,
            lambda: "each sample must contain at least one observed timestep",
        )
        safe = torch.where(observed, signal, torch.zeros_like(signal))
        torch._check_tensor_all(
            torch.isfinite(safe),
            lambda: "observed values must be finite",
        )
        stats_dtype = (
            torch.float32 if signal.dtype in (torch.float16, torch.bfloat16) else signal.dtype
        )
        stats = safe.to(stats_dtype)
        stats_count = count.to(stats_dtype)
        mean = stats.sum(dim=2, keepdim=True) / stats_count
        centered = torch.where(observed, signal.to(stats_dtype) - mean, 0.0)
        scale = (centered.square().sum(dim=2, keepdim=True) / stats_count).sqrt()
        minimum = max(self.eps, torch.finfo(stats_dtype).tiny)
        normalized = torch.where(observed, centered / scale.clamp_min(minimum), 0.0).to(
            values.dtype
        )
        if self.observed_channel is None:
            return normalized
        return torch.cat((normalized[:, :index], validity, normalized[:, index:]), dim=1)


__all__ = ["InstanceStandardize", "Standardize"]
