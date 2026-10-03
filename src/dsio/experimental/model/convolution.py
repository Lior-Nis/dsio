"""Dense convolutional signal blocks that preserve every timestep."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DenseConv1d(nn.Module):
    """Apply two same-length convolutions without pooling.

    Consumes:
        A tensor ``[batch, channels, time]``. When ``observed_channel`` is set, one
        additional channel contains shared zero-or-one timestep validity.

    Produces:
        A tensor ``[batch, output, time]``; the temporal extent is unchanged.

    Parameters:
        ``channels``: input channels. ``output``: output channels. ``hidden``: width of
        the hidden convolution (default 16). ``kernel_size``: odd convolution kernel
        (default 5), padded symmetrically to preserve time. ``observed_channel``:
        optional index of the structural validity channel; it is not a learned feature.

    Devices:
        CPU and accelerators. Inputs are cast to the convolution parameters' dtype.

    Limitations:
        Exactly Conv1d -> ReLU -> Conv1d. With validity, hidden and output padding are
        zeroed so real predictions cannot depend on co-batched lengths. There is no
        pooling, residual path, or causal padding.

    Example:
        >>> import torch
        >>> model = DenseConv1d(channels=3, output=2)
        >>> tuple(model(torch.ones(4, 3, 9)).shape)
        (4, 2, 9)
    """

    def __init__(
        self,
        channels: int,
        output: int,
        hidden: int = 16,
        kernel_size: int = 5,
        observed_channel: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = _positive("channels", channels)
        hidden = _positive("hidden", hidden)
        output = _positive("output", output)
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 1
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be an odd positive integer")
        if observed_channel is not None and (
            isinstance(observed_channel, bool) or not isinstance(observed_channel, int)
        ):
            raise ValueError("observed_channel must be an integer or None")
        input_channels = self.channels + 1
        if (
            observed_channel is not None
            and not -input_channels <= observed_channel < input_channels
        ):
            raise ValueError(
                f"observed_channel {observed_channel} must identify one of "
                f"{input_channels} input channels"
            )
        self.observed_channel = observed_channel
        padding = kernel_size // 2
        self.input = nn.Conv1d(
            self.channels,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.output = nn.Conv1d(hidden, output, kernel_size=kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"channel-first signal must have rank 3, got shape {tuple(x.shape)}")
        expected_channels = self.channels + (self.observed_channel is not None)
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"channel axis 1 expected {expected_channels}, got {x.shape[1]} "
                f"in shape {tuple(x.shape)}"
            )
        if x.shape[2] < 1:
            raise ValueError("time axis 2 must be non-empty")
        dtype = next(self.parameters()).dtype
        values = x if x.dtype == dtype else x.to(dtype)
        observed = None
        if self.observed_channel is not None:
            index = self.observed_channel % expected_channels
            validity = values[:, index : index + 1]
            torch._check_tensor_all(
                torch.isfinite(validity) & ((validity == 0) | (validity == 1)),
                lambda: "observed validity channel must contain only zero or one",
            )
            observed = validity.bool()
            values = torch.cat((values[:, :index], values[:, index + 1 :]), dim=1)
        hidden = self.activation(self.input(values))
        if observed is not None:
            hidden = torch.where(observed, hidden, 0.0)
        result = self.output(hidden)
        return result if observed is None else torch.where(observed, result, 0.0)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["DenseConv1d"]
