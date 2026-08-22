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

from dsio.model.registry import augmentor, backbone, head, loss, preprocessor, transform


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


@head("mae_decoder")
def mae_decoder_head(in_dim: int, channels: int, length: int, hidden_mult: int = 2) -> nn.Module:
    """MAE's reconstruction head: predict every position of the original window, back in
    the ``[channels, length]`` shape the input arrived in.

    Registered like any other head, moved here from the deleted ``ssl.methods.
    MaskedReconstruction.build_head`` — ``ssl_task.py`` now builds it through ``HEADS`` and
    ``_accepted`` exactly the way ``torch_task.py`` builds a classification head, which is
    what let the pretext objective stop being a separate kind of thing that builds its own
    head. Its output only makes sense paired with :class:`MaskedMSE` and a masked
    :class:`~dsio.dataset.dataset.WindowDataset` target, which is why
    :func:`~dsio.model.module.export_encoder` never ships it with the encoder.
    """
    return nn.Sequential(
        nn.Linear(in_dim, in_dim * hidden_mult),
        nn.GELU(),
        nn.Linear(in_dim * hidden_mult, channels * length),
        nn.Unflatten(-1, (channels, length)),
    )


@head("simclr_projector")
def simclr_projector_head(in_dim: int, out_dim: int = 64) -> nn.Module:
    """SimCLR's projection head: NT-Xent compares windows in this space, not the encoder's
    own feature space — moved here from the deleted ``ssl.methods.SimCLR.build_head``."""
    return nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))


@head("vicreg_projector")
def vicreg_projector_head(in_dim: int, out_dim: int = 64) -> nn.Module:
    """VICReg's projection head, moved here from the deleted ``ssl.methods.VICReg.
    build_head``. The ``BatchNorm1d`` matters: :class:`VICReg`'s variance term assumes a
    projector that does not itself normalise away the collapse it exists to detect."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim * 2),
        nn.BatchNorm1d(in_dim * 2),
        nn.ReLU(),
        nn.Linear(in_dim * 2, out_dim),
    )


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


@loss("masked_mse")
class MaskedMSE(nn.Module):
    """Reconstruction loss for a target that carries NaN outside masked positions.

    NaN is the continuous analogue of MLM's ``-100``: :class:`~dsio.dataset.dataset.WindowDataset`
    writes the original value at every position its mask hid and NaN everywhere the model
    was allowed to see the input, so this is the whole mechanism that keeps a masked
    autoencoder from winning by copying — a reconstruction that only matches the visible
    input is never compared against anything there, because there is nothing there to
    compare against.

    Selecting ``~torch.isnan(target)`` **before** computing ``(prediction - target) ** 2``
    is required, not stylistic: a NaN that reaches the subtraction produces a NaN error,
    and ``nan * 0`` is still NaN, so masking after the arithmetic would poison the mean
    (and the gradient) regardless of which positions were meant to be excluded.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(target)
        if not valid.any():
            raise ValueError(
                "target has no masked positions to reconstruct; the mask hid nothing"
            )
        return nn.functional.mse_loss(prediction[valid], target[valid])

    def diagnostics(
        self, prediction: torch.Tensor, target: torch.Tensor, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The copy-vs-learned split: ADR 0011's detector for a model that is copying.

        ``masked_mse`` is exactly what :meth:`forward` scores. ``visible_mse`` needs the
        true value at every *visible* position, which the sentinel target no longer
        carries — ``x`` does: masking only zeroes the hidden positions, so every visible
        one still holds the original signal. If ``visible_mse`` collapses while
        ``masked_mse`` does not, the model is copying rather than reconstructing.

        Called from :meth:`~dsio.model.module.DsioModule._common_step`, on the same
        ``prediction`` that step already computed — no extra forward pass, unlike the
        first attempt at restoring this diagnostic via a training-batch-end hook.
        """
        hidden = ~torch.isnan(target)
        visible = ~hidden
        return {
            "masked_mse": nn.functional.mse_loss(prediction[hidden], target[hidden]),
            "visible_mse": nn.functional.mse_loss(prediction[visible], x[visible]),
        }


def _pair_halves(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a two-view batch into its matched halves, from the pair-index target alone.

    ``target[i]`` names ``i``'s partner — the same window, a second independent
    augmentation — and the relation is symmetric: ``target[target[i]] == i``. Taking the
    smaller index of each pair gives one representative per match, and indexing ``target``
    at those positions gives the other, which recovers the two view-halves a loss like
    :class:`VICReg` needs without assuming how the batch is laid out (a contiguous "first
    half / second half", interleaved, or anything else) — only the index relationship
    :class:`~dsio.dataset.dataset.TwoViewCollate` promises.
    """
    order = torch.arange(target.shape[0], device=target.device)
    first = (order < target).nonzero(as_tuple=True)[0]
    return prediction[first], prediction[target[first]]


@loss("nt_xent")
class NTXent(nn.Module):
    """SimCLR's contrastive loss, over a batch :class:`~dsio.dataset.dataset.TwoViewCollate` built.

    ``prediction`` is the whole batch's projected embeddings — ``2 * batch`` rows, two per
    window — and ``target`` is each row's pair index, exactly the ``(arange(2 * batch) +
    batch) % (2 * batch)`` computation the deleted ``ssl.methods.SimCLR.step()`` used to do
    inline. Every row that is not a window's own partner is a negative, so the batch size is
    part of the objective rather than a performance knob: halving it changes what is being
    optimised, which is worth stating because dsio's cache treats batch size as speed-only
    for every other purpose.

    This is the whole of what ``SimCLR.step()`` used to do: with the views and the pair
    index already built by the time a loss sees them, NT-Xent needs nothing but
    ``(prediction, target)`` — no batch dict, no subclass.
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[0] < 4:
            raise ValueError(
                "nt_xent needs at least two windows per view; the batch has no negatives"
            )
        z = nn.functional.normalize(prediction, dim=-1)
        similarity = (z @ z.T) / self.temperature
        similarity.fill_diagonal_(float("-inf"))
        return nn.functional.cross_entropy(similarity, target)

    def diagnostics(
        self, prediction: torch.Tensor, target: torch.Tensor, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``mean_abs_cosine``: SimCLR's collapse tell. A collapsed encoder maps every
        window to the same point; the loss does not obviously misbehave, but the mean
        off-diagonal cosine similarity goes to 1.0. ``nt_xent`` is repeated here under its
        own name, identical to ``{stage}/loss`` — that is the metric name earlier
        pretraining runs logged, and dropping it would silently break a dashboard built
        against it.
        """
        with torch.no_grad():
            z = nn.functional.normalize(prediction, dim=-1)
            similarity = (z @ z.T) / self.temperature
            similarity.fill_diagonal_(float("-inf"))
            nt_xent = nn.functional.cross_entropy(similarity, target)
            n = z.shape[0]
            off_diagonal = (z @ z.T).fill_diagonal_(0.0).abs().sum() / (n * (n - 1))
        return {"nt_xent": nt_xent, "mean_abs_cosine": off_diagonal}


@loss("vicreg")
class VICReg(nn.Module):
    """Variance-Invariance-Covariance regularisation: no negatives, no momentum encoder —
    collapse is prevented by an explicit variance term instead.

    Reads the same pair-index ``target`` :class:`NTXent` does, but only to recover which two
    rows are one window's pair (:func:`_pair_halves`); the variance and covariance terms are
    then per-view marginal statistics computed from ``prediction`` alone, and the invariance
    term is an indexed MSE over the two halves the pair index selects. That is what makes
    ``(prediction, target)`` sufficient here too — nothing this needs lives outside the
    batch of predictions and the index naming each row's partner.
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.target_std = target_std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        za, zb = _pair_halves(prediction, target)
        if za.shape[0] < 2:
            raise ValueError("vicreg needs at least two pairs to estimate variance")
        invariance = nn.functional.mse_loss(za, zb)
        variance = 0.5 * (self._variance(za) + self._variance(zb))
        covariance = 0.5 * (self._covariance(za) + self._covariance(zb))
        return (
            self.sim_weight * invariance
            + self.var_weight * variance
            + self.cov_weight * covariance
        )

    def diagnostics(
        self, prediction: torch.Tensor, target: torch.Tensor, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``embedding_std`` is the number to watch: if it sits below ``target_std``, the
        representation is collapsing regardless of what the total loss is doing."""
        za, zb = _pair_halves(prediction, target)
        with torch.no_grad():
            invariance = nn.functional.mse_loss(za, zb)
            variance = 0.5 * (self._variance(za) + self._variance(zb))
            covariance = 0.5 * (self._covariance(za) + self._covariance(zb))
            std = za.std(dim=0).mean()
        return {
            "invariance": invariance,
            "variance_penalty": variance,
            "covariance": covariance,
            "embedding_std": std,
        }

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.relu(self.target_std - std).mean()

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        centred = z - z.mean(dim=0)
        cov = (centred.T @ centred) / (z.shape[0] - 1)
        off_diagonal = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
        return off_diagonal / z.shape[1]


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


# --- view augmentors, for two-view contrastive collation (DsioModule itself has no ------
# --- stochastic slot; see TwoViewCollate in nn/data.py) ---------------------------------


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
