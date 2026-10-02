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
    are constants (``mean=[0]``, ``scale=[255]`` for 8-bit pixels). Integer inputs are cast
    to float32 first, so the same stage serves raw predictor inputs and float training
    batches.

    Consumes:
        A tensor whose ``axis`` dimension holds the features; any other dimensions
        broadcast. A single-value ``mean``/``scale`` applies to every feature.

    Produces:
        A float32 tensor of the input's shape: ``(x - mean) / scale``.

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
        values = x if x.is_floating_point() else x.float()
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


__all__ = ["Standardize"]
