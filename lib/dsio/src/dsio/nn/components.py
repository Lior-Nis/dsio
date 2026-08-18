"""Concrete components: enough to train something real, few enough to read in one sitting.

The spine ships a small set deliberately. A project registers its own — that is what the
registries are for — and dsio's job is to make the slot exist and be wired correctly, not
to be a model zoo it then has to maintain.

Every module here takes ``[batch, channels, time]`` and is shape-checked on the way in,
because a silent broadcast between a channels-first and a channels-last tensor produces a
model that trains, converges to something, and is wrong.
"""

from __future__ import annotations

import torch
from torch import nn

from dsio.nn.registry import augmentor, backbone, head, loss, preprocessor, transform


def _check_3d(x: torch.Tensor, who: str) -> None:
    if x.ndim != 3:
        raise ValueError(
            f"{who} expects [batch, channels, time], got shape {tuple(x.shape)}; "
            "a 2-D tensor here usually means the collate dropped the channel axis"
        )


# --- backbones ----------------------------------------------------------------------


@backbone("mlp1d")
class MLP1d(nn.Module):
    """Flatten and project. The baseline every other backbone must beat."""

    def __init__(self, channels: int, length: int, hidden: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * length, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "MLP1d")
        return self.net(x)


@backbone("conv1d")
class Conv1dEncoder(nn.Module):
    """Strided dilated convolutions with global pooling.

    Pools over time rather than flattening, so the same weights accept any window length.
    A backbone whose parameter count depends on the window length forces a retrain for
    every change to a view — which is exactly the coupling the index layer removed.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        out_dim: int = 64,
        depth: int = 3,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = channels
        for level in range(depth):
            dilation = 2**level
            layers += [
                nn.Conv1d(
                    in_channels,
                    hidden,
                    kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            ]
            in_channels = hidden
        self.body = nn.Sequential(*layers)
        self.project = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "Conv1dEncoder")
        pooled = self.body(x).mean(dim=-1)
        return self.project(pooled)


# --- heads --------------------------------------------------------------------------


@head("linear")
def linear_head(in_dim: int = 64, out_dim: int = 2) -> nn.Module:
    return nn.Linear(in_dim, out_dim)


@head("mlp")
def mlp_head(in_dim: int = 64, hidden: int = 64, out_dim: int = 2) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


@head("identity")
def identity_head() -> nn.Module:
    """For pretraining, where the loss consumes features directly."""
    return nn.Identity()


# --- losses -------------------------------------------------------------------------


@loss("cross_entropy")
class CrossEntropy(nn.Module):
    """Cross-entropy that accepts either hard integer labels or soft ratios.

    A windowed label is often a *ratio* — what fraction of the window was positive — and
    thresholding it to fit a loss throws away the distinction between a window that is 51%
    positive and one that is 99%. Accepting both is what lets a label policy stay a data
    decision rather than becoming a loss decision.
    """

    # Declared so mypy sees a tensor: register_buffer is typed as returning Tensor | Module.
    weight: torch.Tensor | None

    def __init__(self, weight: list[float] | None = None, threshold: float | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "weight", None if weight is None else torch.tensor(weight, dtype=torch.float32)
        )
        self.threshold = threshold

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.is_floating_point():
            if self.threshold is None:
                raise ValueError(
                    "cross_entropy received soft targets but has no threshold; set one to "
                    "binarise, or use a loss that consumes ratios directly"
                )
            target = (target > self.threshold).long()
        return nn.functional.cross_entropy(prediction, target.long(), weight=self.weight)


@loss("bce")
def bce_loss() -> nn.Module:
    class _BCE(nn.Module):
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return nn.functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1), target.float()
            )

    return _BCE()


@loss("mse")
def mse_loss() -> nn.Module:
    class _MSE(nn.Module):
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return nn.functional.mse_loss(prediction.squeeze(-1), target.float())

    return _MSE()


# --- transforms and preprocessors ---------------------------------------------------


@transform("identity")
def identity_transform() -> nn.Module:
    return nn.Identity()


@transform("instance_standardize")
class InstanceStandardize(nn.Module):
    """Per-window, per-channel standardisation.

    Deliberately *not* a fitted preprocessor: its statistics come from the window itself,
    so it cannot leak anything from the training set into a test window. Where corpus-level
    statistics are wanted, they belong in a preprocessor fitted on the train fold — which
    is a different slot on purpose.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "InstanceStandardize")
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)


@preprocessor("fixed_standardize")
class FixedStandardize(nn.Module):
    """Standardise by statistics supplied from outside — fitted on the train fold only."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: list[float], std: list[float], eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "FixedStandardize")
        return (x - self.mean) / (self.std + self.eps)


# --- augmentors ---------------------------------------------------------------------


@augmentor("jitter")
class Jitter(nn.Module):
    """Additive Gaussian noise, scaled per channel by that channel's own spread.

    A fixed sigma means the same augmentation is negligible on one sensor and destroys
    another; scaling by the observed spread keeps its strength comparable across channels
    whose units have nothing to do with each other.
    """

    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "Jitter")
        scale = x.std(dim=-1, keepdim=True) * self.sigma
        return x + torch.randn_like(x) * scale


@augmentor("random_scale")
class RandomScale(nn.Module):
    """Multiply each channel by a random gain, for amplitude-invariant features."""

    def __init__(self, low: float = 0.9, high: float = 1.1) -> None:
        super().__init__()
        if low > high:
            raise ValueError(f"low {low} must not exceed high {high}")
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_3d(x, "RandomScale")
        gain = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).uniform_(
            self.low, self.high
        )
        return x * gain


@augmentor("none")
def no_augmentation() -> nn.Module:
    return nn.Identity()
