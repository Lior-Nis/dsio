"""Pretext objectives, one per family, over the shared component chain.

Three methods rather than seven, chosen to span the families rather than to be exhaustive:

- **MAE** — generative. Hide part of the input, reconstruct it.
- **SimCLR** — contrastive. Pull two views of the same window together, push others apart.
- **VICReg** — redundancy reduction. No negatives at all; collapse is prevented by an
  explicit variance term.

SimCLR and VICReg are each a small object with one method, ``step(module, x) -> (loss,
logs)``, so the encoder, the augmentations and the training loop are shared and only the
objective differs. MAE has moved off that shape: masking now lives on the dataset and the
loss reads a NaN-sentinel target directly, so the generic ``(prediction, target)`` chain in
``DsioModule._common_step`` is MAE's whole training step, and ``MaskedReconstruction``
exists only to build its decoder head.

**Every method logs the diagnostic that reveals its own failure mode**, because in SSL the
loss does not. A collapsed SimCLR encoder that maps everything to one point has a *low*
loss and useless features; a VICReg run whose variance term has been overwhelmed looks like
smooth convergence. Those numbers are logged next to the loss, not left to be discovered
when a downstream probe fails weeks later.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn

from dsio.config.registry import Registry

METHODS: Registry[type] = Registry("ssl_method")


def ssl_method(name: str):  # type: ignore[no-untyped-def]
    """Register a pretext objective."""
    return METHODS.register(name)


class PretextObjective(Protocol):
    """What every pretext objective must provide, MAE included.

    Narrower than :class:`SslMethod`: MAE builds a head but no longer implements ``step`` —
    its training step is the generic ``(prediction, target)`` chain, driven by the
    dataset's mask and a mask-aware loss, not by an objective-specific call. This is the
    type :class:`~dsio.ssl.module.SslModule` accepts, since it no longer calls ``step`` on
    anything itself.
    """

    def build_head(self, feature_dim: int, channels: int, length: int) -> nn.Module:
        """The objective's own output layer, which is discarded after pretraining."""
        ...


class SslMethod(PretextObjective, Protocol):
    """What a *contrastive* pretext objective must provide, on top of a head.

    SimCLR and VICReg still drive their own step: they augment two views of ``x`` and
    compute a loss no ``(prediction, target)`` pair could express, so they keep the
    explicit ``step`` call that :class:`~dsio.ssl.module.ContrastiveModule` reaches for.
    """

    def step(self, module: Any, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the loss and any scalars worth logging alongside it."""
        ...


@ssl_method("mae")
class MaskedReconstruction:
    """Masked autoencoding: hide part of the signal, predict it from what remains.

    The mask itself lives on the training dataset now (``WindowDataset(..., mask=...)``),
    which also writes the NaN sentinel a mask-aware loss needs — see
    :class:`~dsio.nn.components.MaskedMSE`. That makes ``(x_masked, target)`` exactly the
    ``(prediction, target)`` shape the generic chain already knows how to consume, so this
    class no longer implements ``step``; it exists to build the reconstruction head, which
    is method-specific in a way masking and loss selection are not.
    """

    def build_head(self, feature_dim: int, channels: int, length: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, channels * length),
            nn.Unflatten(-1, (channels, length)),
        )


@ssl_method("simclr")
class SimCLR:
    """NT-Xent over two augmented views of the same window.

    The other samples in the batch are the negatives, so the batch size is part of the
    objective rather than a performance knob — halving it changes what is being optimised.
    That is worth stating because dsio's cache treats batch size as speed-only for every
    other purpose.
    """

    def __init__(self, augment: nn.Module, temperature: float = 0.2, projection_dim: int = 64):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.augment = augment
        self.temperature = temperature
        self.projection_dim = projection_dim

    def build_head(self, feature_dim: int, channels: int, length: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, self.projection_dim),
        )

    def step(self, module: Any, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        batch = x.shape[0]
        if batch < 2:
            raise ValueError("SimCLR needs at least two samples; the batch has no negatives")

        z = nn.functional.normalize(
            module.head(module.encode(torch.cat([self.augment(x), self.augment(x)], dim=0))),
            dim=-1,
        )
        similarity = (z @ z.T) / self.temperature
        similarity.fill_diagonal_(float("-inf"))
        # View i's positive is view i+batch, and vice versa.
        targets = torch.arange(2 * batch, device=x.device)
        targets = (targets + batch) % (2 * batch)
        loss = nn.functional.cross_entropy(similarity, targets)

        with torch.no_grad():
            # Mean off-diagonal cosine similarity. Approaching 1.0 is collapse: every
            # window maps to the same point, the loss looks fine, the features are useless.
            off_diagonal = (z @ z.T).fill_diagonal_(0.0).abs().sum() / (
                2 * batch * (2 * batch - 1)
            )
        return loss, {"nt_xent": float(loss.detach()), "mean_abs_cosine": float(off_diagonal)}


@ssl_method("vicreg")
class VICReg:
    """Variance-Invariance-Covariance regularisation: no negatives, no momentum encoder.

    The variance term is what prevents collapse, which makes it the one number worth
    watching: if the per-dimension standard deviation of the embedding sits below the
    target, the representation is collapsing regardless of what the total loss is doing.
    """

    def __init__(
        self,
        augment: nn.Module,
        projection_dim: int = 64,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        target_std: float = 1.0,
    ) -> None:
        self.augment = augment
        self.projection_dim = projection_dim
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.target_std = target_std

    def build_head(self, feature_dim: int, channels: int, length: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, self.projection_dim),
        )

    def step(self, module: Any, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        batch = x.shape[0]
        if batch < 2:
            raise ValueError("VICReg needs at least two samples to estimate variance")

        za = module.head(module.encode(self.augment(x)))
        zb = module.head(module.encode(self.augment(x)))

        invariance = nn.functional.mse_loss(za, zb)
        variance = 0.5 * (self._variance(za) + self._variance(zb))
        covariance = 0.5 * (self._covariance(za) + self._covariance(zb))

        loss = (
            self.sim_weight * invariance
            + self.var_weight * variance
            + self.cov_weight * covariance
        )
        with torch.no_grad():
            std = za.std(dim=0).mean()
        return loss, {
            "invariance": float(invariance.detach()),
            "variance_penalty": float(variance.detach()),
            "covariance": float(covariance.detach()),
            "embedding_std": float(std),
        }

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.relu(self.target_std - std).mean()

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        centred = z - z.mean(dim=0)
        cov = (centred.T @ centred) / (z.shape[0] - 1)
        off_diagonal = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
        return off_diagonal / z.shape[1]
