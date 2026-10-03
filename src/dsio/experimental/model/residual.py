"""Models that learn a bounded correction around an explicit baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BaselineResidual(nn.Module):
    """Add a zero-initialized bounded pointwise residual to a baseline channel.

    Consumes:
        A floating tensor ``[batch, points, features]``. ``baseline_channel`` supplies
        the prediction to improve and ``validity_channel`` contains exactly zero or one.

    Produces:
        ``[batch, points]``. Initialization is exactly the valid baseline; invalid rows
        are zero even when their other feature values are non-finite.

    Parameters:
        ``features``: input width. ``baseline_channel`` and ``validity_channel``: distinct
        non-negative feature indices. ``hidden``: pointwise residual width (default 32).
        ``bound``: maximum absolute correction around the baseline (default 0.01).

    Devices:
        CPU and accelerators. The residual network uses its parameters' floating dtype;
        the output preserves the input dtype and device so initialization remains an exact
        identity around the declared baseline. Input and parameters must share a device.

    Limitations:
        The corrector is exactly Linear -> ReLU -> Linear. It is pointwise and bounded by
        ``tanh``; temporal context and alternative residual networks are outside this block.

    Example:
        >>> import torch
        >>> model = BaselineResidual(features=3, baseline_channel=1, validity_channel=2)
        >>> x = torch.tensor([[[4.0, 2.0, 1.0], [9.0, 3.0, 0.0]]])
        >>> model(x).tolist()
        [[2.0, 0.0]]
    """

    _bound: Tensor

    def __init__(
        self,
        features: int,
        baseline_channel: int,
        validity_channel: int,
        hidden: int = 32,
        bound: float = 0.01,
    ) -> None:
        super().__init__()
        self.features = _positive("features", features)
        hidden = _positive("hidden", hidden)
        self.baseline_channel = _channel("baseline_channel", baseline_channel, features)
        self.validity_channel = _channel("validity_channel", validity_channel, features)
        if baseline_channel == validity_channel:
            raise ValueError("baseline_channel and validity_channel must be distinct")
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"bound must be a finite positive number, got {bound!r}")
        self.bound = float(bound)
        if not math.isfinite(self.bound) or self.bound <= 0:
            raise ValueError(f"bound must be a finite positive number, got {bound!r}")
        self.residual = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        bound_tensor = torch.tensor(self.bound, dtype=final.weight.dtype)
        if not torch.isfinite(bound_tensor).item() or bound_tensor.item() <= 0:
            raise ValueError(
                f"bound must be representable in the model dtype {final.weight.dtype}, "
                f"got {bound!r}"
            )
        self.register_buffer("_bound", bound_tensor, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"BaselineResidual expects rank 3 [batch, points, features], "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[0] < 1:
            raise ValueError("batch axis 0 must be non-empty")
        if x.shape[1] < 1:
            raise ValueError("point axis 1 must be non-empty")
        if x.shape[2] != self.features:
            raise ValueError(
                f"feature axis 2 expected {self.features}, got {x.shape[2]} "
                f"in shape {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError(f"BaselineResidual expects a floating tensor, got {x.dtype}")

        parameter = next(self.parameters())
        if x.device != parameter.device:
            raise ValueError(
                f"input and BaselineResidual parameters must share a device, "
                f"got {x.device} and {parameter.device}"
            )

        validity = x[:, :, self.validity_channel]
        binary = (validity == 0) | (validity == 1)
        valid = validity.bool()
        safe = torch.where(valid.unsqueeze(-1), x, 0.0)
        torch._assert_async(
            binary.all() & torch.isfinite(safe).all(),
            "invalid residual input: validity channel must contain only zero or one; "
            "observed features must be finite",
        )
        baseline = safe[:, :, self.baseline_channel]
        dtype = parameter.dtype
        network_input = safe if safe.dtype == dtype else safe.to(dtype)
        torch._assert_async(
            torch.isfinite(network_input).all() & torch.isfinite(self._bound) & (self._bound > 0),
            "residual input or bound is not representable in the model dtype",
        )
        logits = self.residual(network_input).squeeze(-1)
        torch._assert_async(
            torch.isfinite(logits).all(),
            "residual network produced non-finite values",
        )
        correction = self._bound * torch.tanh(logits)
        return torch.where(valid, baseline + correction.to(baseline.dtype), 0.0)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _channel(name: str, value: int, features: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value < features:
        raise ValueError(f"{name} must be in [0, {features}), got {value}")
    return value


__all__ = ["BaselineResidual"]
