"""Pretext objectives, and the collapse diagnostics that the loss cannot show."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from torch import nn  # noqa: E402

from dsio.nn.components import Conv1dEncoder, Jitter  # noqa: E402
from dsio.ssl import (  # noqa: E402
    METHODS,
    MaskedReconstruction,
    SimCLR,
    SslModule,
    VICReg,
)
from dsio.ssl.masking import PatchMask, SpanMask  # noqa: E402

CHANNELS, LENGTH, DIM = 2, 128, 16


def module_for(method) -> SslModule:  # type: ignore[no-untyped-def]
    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    return SslModule(
        method=method,
        backbone=backbone,
        head=method.build_head(DIM, CHANNELS, LENGTH),
        loss=nn.Identity(),
    )


@pytest.fixture
def signal() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, CHANNELS, LENGTH)


# --- MAE ------------------------------------------------------------------------------


def test_mae_produces_a_finite_loss_and_its_diagnostics(signal: torch.Tensor) -> None:
    method = MaskedReconstruction(mask=SpanMask(0.5, span=16))
    loss, logs = method.step(module_for(method), signal)
    assert torch.isfinite(loss)
    assert {"masked_mse", "visible_mse"} == set(logs)


def test_mae_scores_only_the_masked_positions(signal: torch.Tensor) -> None:
    """The failure this prevents: averaging over every position lets the model score well
    by copying the visible input, which it will, because copying is easier than inferring.

    A predictor that reproduces the target perfectly where it can see it and outputs zero
    where it cannot must still be penalised.
    """
    method = MaskedReconstruction(mask=PatchMask(0.5, patch=16), norm_target=False)
    module = module_for(method)
    mask = PatchMask(0.5, patch=16)(signal)

    class Copier(nn.Module):
        """Perfect on visible positions, useless on hidden ones."""

        def forward(self, _: torch.Tensor) -> torch.Tensor:
            out = signal.clone()
            out[mask.unsqueeze(1).expand_as(out)] = 0.0
            return out

    module.head = Copier()
    module.encode = lambda x: x  # type: ignore[method-assign]
    method.mask = lambda x, generator=None: mask  # type: ignore[assignment]

    loss, logs = method.step(module, signal)
    assert logs["visible_mse"] == pytest.approx(0.0, abs=1e-6)
    assert loss.item() > 0.1, "copying the visible input must not score well"


def test_mae_normalises_the_target_by_default(signal: torch.Tensor) -> None:
    """Reconstructing raw amplitude makes the loss dominated by the loudest channel, so the
    model spends its capacity on whichever sensor has the largest units."""
    loud = signal.clone()
    loud[:, 0] *= 100.0
    method = MaskedReconstruction(mask=SpanMask(0.5, span=16), norm_target=True)
    module = module_for(method)
    torch.manual_seed(0)
    quiet_loss, _ = method.step(module, signal)
    torch.manual_seed(0)
    loud_loss, _ = method.step(module, loud)
    assert loud_loss.item() < quiet_loss.item() * 50


# --- SimCLR ---------------------------------------------------------------------------


def test_simclr_produces_a_finite_loss(signal: torch.Tensor) -> None:
    method = SimCLR(augment=Jitter(0.2))
    loss, logs = method.step(module_for(method), signal)
    assert torch.isfinite(loss)
    assert "mean_abs_cosine" in logs


def test_simclr_reports_collapse_that_the_loss_hides(signal: torch.Tensor) -> None:
    """A collapsed encoder maps everything to one point. Mean cosine goes to 1; the loss
    does not obviously misbehave, and the features are useless."""
    method = SimCLR(augment=Jitter(0.2))
    module = module_for(method)
    module.head = nn.Sequential(nn.Linear(DIM, 8), _Constant())
    _, logs = method.step(module, signal)
    assert logs["mean_abs_cosine"] == pytest.approx(1.0, abs=1e-3)


def test_simclr_needs_negatives(signal: torch.Tensor) -> None:
    method = SimCLR(augment=Jitter(0.2))
    with pytest.raises(ValueError, match="no negatives"):
        method.step(module_for(method), signal[:1])


def test_simclr_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SimCLR(augment=Jitter(0.2), temperature=0.0)


def test_simclr_loss_falls_when_views_agree(signal: torch.Tensor) -> None:
    """Sanity on the objective itself: identical views of distinct samples is the easy case."""
    weak = SimCLR(augment=nn.Identity(), temperature=0.2)
    strong = SimCLR(augment=Jitter(3.0), temperature=0.2)
    module = module_for(weak)
    torch.manual_seed(0)
    easy, _ = weak.step(module, signal)
    torch.manual_seed(0)
    hard, _ = strong.step(module, signal)
    assert easy.item() < hard.item()


# --- VICReg ---------------------------------------------------------------------------


def test_vicreg_produces_a_finite_loss_and_its_terms(signal: torch.Tensor) -> None:
    method = VICReg(augment=Jitter(0.2))
    loss, logs = method.step(module_for(method), signal)
    assert torch.isfinite(loss)
    assert {"invariance", "variance_penalty", "covariance", "embedding_std"} == set(logs)


def test_vicreg_penalises_a_collapsed_embedding(signal: torch.Tensor) -> None:
    """The variance term is what prevents collapse, so it must actually fire on one."""
    method = VICReg(augment=Jitter(0.2), target_std=1.0)
    module = module_for(method)
    module.head = nn.Sequential(nn.Linear(DIM, 8), _Constant())
    _, logs = method.step(module, signal)
    # 0.99 rather than 1.0: the term is relu(target - sqrt(var + 1e-4)), and sqrt(1e-4)
    # is 0.01. The epsilon is what keeps the gradient finite at exactly zero variance.
    assert logs["variance_penalty"] == pytest.approx(0.99, abs=1e-3)
    assert logs["embedding_std"] == pytest.approx(0.0, abs=1e-4)


def test_vicreg_needs_more_than_one_sample(signal: torch.Tensor) -> None:
    method = VICReg(augment=Jitter(0.2))
    with pytest.raises(ValueError, match="estimate variance"):
        method.step(module_for(method), signal[:1])


# --- shared -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        MaskedReconstruction(mask=SpanMask(0.5, span=16)),
        SimCLR(augment=Jitter(0.2)),
        VICReg(augment=Jitter(0.2)),
    ],
    ids=["mae", "simclr", "vicreg"],
)
def test_every_method_trains_the_same_encoder(method, signal: torch.Tensor) -> None:
    """The property that makes encoders interchangeable downstream: the backbone does not
    know which objective is training it."""
    module = module_for(method)
    loss, _ = method.step(module, signal)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.backbone.parameters())


@pytest.mark.parametrize(
    "method",
    [
        MaskedReconstruction(mask=SpanMask(0.5, span=16)),
        SimCLR(augment=Jitter(0.2)),
        VICReg(augment=Jitter(0.2)),
    ],
    ids=["mae", "simclr", "vicreg"],
)
def test_the_exported_encoder_excludes_the_objective_head(method, signal: torch.Tensor) -> None:
    """A decoder trained to reconstruct masked spans has no meaning outside the pretext
    task, and shipping it invites someone to load it as part of the model."""
    state = module_for(method).encoder_state()
    assert state and all(not key.startswith("head.") for key in state)
    assert any(key.startswith("backbone.") for key in state)


def test_methods_are_registered() -> None:
    assert {"mae", "simclr", "vicreg"} <= set(METHODS.names())


class _Constant(nn.Module):
    """Maps every input to the same point: the canonical collapsed representation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)
