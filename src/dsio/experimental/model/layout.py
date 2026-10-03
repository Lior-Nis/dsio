"""Explicit layout adapters for batched multichannel signals."""

from __future__ import annotations

from torch import Tensor, nn


class TimeMajorToChannelFirst(nn.Module):
    """Transpose a time-major signal into the layout expected by PyTorch Conv1d.

    Consumes:
        A rank-three tensor ``[batch, time, channels]``.

    Produces:
        A contiguous tensor ``[batch, channels, time]`` with identical values.

    Parameters:
        ``channels``: required channel extent. ``time``: optional fixed time extent;
        omitted for variable-length padded batches.

    Devices:
        CPU and accelerators; dtype and device are preserved.

    Limitations:
        Rank-three signals only. This module only changes layout.

    Example:
        >>> import torch
        >>> stage = TimeMajorToChannelFirst(channels=2)
        >>> tuple(stage(torch.ones(3, 5, 2)).shape)
        (3, 2, 5)
    """

    def __init__(self, channels: int, time: int | None = None) -> None:
        super().__init__()
        self.channels = _extent("channels", channels)
        self.time = None if time is None else _extent("time", time)

    def forward(self, x: Tensor) -> Tensor:
        _shape(x, layout="time-major", channel_axis=2, channels=self.channels, time=self.time)
        return x.transpose(1, 2).contiguous()


class ChannelFirstToTimeMajor(nn.Module):
    """Transpose a channel-first signal into a public time-major output.

    Consumes:
        A rank-three tensor ``[batch, channels, time]``.

    Produces:
        A contiguous tensor ``[batch, time, channels]`` with identical values.

    Parameters:
        ``channels``: required channel extent. ``time``: optional fixed time extent;
        omitted for variable-length padded batches.

    Devices:
        CPU and accelerators; dtype and device are preserved.

    Limitations:
        Rank-three signals only. This module only changes layout.

    Example:
        >>> import torch
        >>> stage = ChannelFirstToTimeMajor(channels=2)
        >>> tuple(stage(torch.ones(3, 2, 5)).shape)
        (3, 5, 2)
    """

    def __init__(self, channels: int, time: int | None = None) -> None:
        super().__init__()
        self.channels = _extent("channels", channels)
        self.time = None if time is None else _extent("time", time)

    def forward(self, x: Tensor) -> Tensor:
        _shape(
            x,
            layout="channel-first",
            channel_axis=1,
            channels=self.channels,
            time=self.time,
        )
        return x.transpose(1, 2).contiguous()


def _extent(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} extent must be a positive integer, got {value!r}")
    return value


def _shape(
    x: Tensor,
    *,
    layout: str,
    channel_axis: int,
    channels: int,
    time: int | None,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"{layout} signal must have rank 3, got shape {tuple(x.shape)}")
    if x.shape[channel_axis] != channels:
        raise ValueError(
            f"channel axis {channel_axis} expected {channels}, got {x.shape[channel_axis]} "
            f"in shape {tuple(x.shape)}"
        )
    time_axis = 1 if channel_axis == 2 else 2
    if time is not None and x.shape[time_axis] != time:
        raise ValueError(
            f"time axis {time_axis} expected {time}, got {x.shape[time_axis]} "
            f"in shape {tuple(x.shape)}"
        )


__all__ = ["ChannelFirstToTimeMajor", "TimeMajorToChannelFirst"]
