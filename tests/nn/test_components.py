"""Registered components: backbones, heads, losses, transforms, augmentors.

The mask-aware loss gets the most scrutiny here, because it is the whole point of Task
6a: a NaN sentinel in the target (see ``WindowDataset`` in ``dsio.nn.data``) is only
useful if a loss that reads it naively actually breaks, and a loss that selects on it
before computing error actually rewards reconstruction over copying.

The contrastive losses and heads below moved here from the deleted
``tests/ssl/test_methods.py`` when Task 6b dissolved ``dsio.ssl``: ``SimCLR``/``VICReg``
are no longer objects with their own ``step()``, they are an ``nt_xent``/``vicreg`` loss
plus a projector head, exactly like every other registered component.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from dsio.nn.components import (  # noqa: E402
    Conv1dEncoder,
    Jitter,
    MaskedMSE,
    NTXent,
    VICReg,
    mae_decoder_head,
    simclr_projector_head,
    vicreg_projector_head,
)
from dsio.nn.data import TwoViewCollate  # noqa: E402
from dsio.nn.module import DsioModule  # noqa: E402
from dsio.nn.registry import AUGMENTORS, HEADS, LOSSES  # noqa: E402

CHANNELS, LENGTH, DIM = 2, 128, 16


def _signal() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A window, a contiguous hidden span, and the two predictions the property compares.

    ``masked_input`` is what a model that merely copies its own (masked) input would
    output: correct at every visible position, and whatever ``apply_mask`` left behind
    (zero) at the hidden ones. ``original`` is what a model that actually reconstructed
    the hidden positions would output.
    """
    torch.manual_seed(0)
    original = torch.randn(4, 2, 32)
    hidden = torch.zeros(4, 32, dtype=torch.bool)
    hidden[:, 10:20] = True  # a contiguous masked span, the MAE shape
    masked_input = original.masked_fill(hidden.unsqueeze(1), 0.0)
    return original, hidden, masked_input, original.clone()


# --- Step 1: the property, and the failure it depends on ---------------------------


def test_whole_window_loss_on_the_sentinel_target_cannot_tell_them_apart() -> None:
    """The failure a mask-aware loss exists to prevent, demonstrated directly.

    The dataset's sentinel target carries the true value at every masked position and NaN
    everywhere else. Feeding that straight into an ordinary whole-window loss — the naive
    thing the generic ``(prediction, target)`` contract would do without this task's
    change — does not merely fail to reward reconstruction over copying; it cannot
    evaluate the comparison at all, because NaN propagates through the arithmetic and
    poisons the mean regardless of what the prediction is. Watch this fail first: it needs
    nothing but ``torch``, no implementation from this task.
    """
    original, hidden, masked_input, reconstruction = _signal()
    sentinel_target = original.masked_fill(~hidden.unsqueeze(1), float("nan"))

    whole_window_mse = nn.functional.mse_loss  # what a plain (prediction, target) loss does
    copy_loss = whole_window_mse(masked_input, sentinel_target)
    reconstruct_loss = whole_window_mse(reconstruction, sentinel_target)

    assert torch.isnan(copy_loss) and torch.isnan(reconstruct_loss), (
        "a whole-window loss over the sentinel target should be poisoned by NaN, not "
        "merely inaccurate -- that is exactly why selecting after the arithmetic doesn't work"
    )
    # The property under test is "reconstruction scores clearly lower than copying". A NaN
    # comparison is always False, so the property does not hold here -- it cannot even be
    # asked of a loss that does not select the sentinel first.
    assert not (reconstruct_loss.item() < copy_loss.item() * 0.01)


def test_masked_mse_scores_reconstruction_below_copying() -> None:
    """The property a mask-aware loss must deliver, once it exists: reconstructing the
    masked positions scores clearly better than copying the (zeroed) visible input, by a
    wide margin rather than a marginal one."""
    original, hidden, masked_input, reconstruction = _signal()
    sentinel_target = original.masked_fill(~hidden.unsqueeze(1), float("nan"))

    loss = MaskedMSE()
    copy_loss = loss(masked_input, sentinel_target)
    reconstruct_loss = loss(reconstruction, sentinel_target)

    assert torch.isfinite(copy_loss)
    assert torch.isfinite(reconstruct_loss)
    assert reconstruct_loss.item() < copy_loss.item() * 0.01, (
        "a model that reconstructs the masked positions must score far better than one "
        "that only copies the (zeroed) visible input"
    )


# --- unit behaviour ------------------------------------------------------------------


def test_masked_mse_ignores_nan_positions_via_autograd() -> None:
    """Selecting before the arithmetic, not after: a NaN that reaches ``(pred - target)
    ** 2`` produces a NaN error, and ``nan * 0`` is still NaN, so masking after the
    arithmetic cannot recover a finite gradient. This is the regression the sentinel
    depends on."""
    prediction = torch.randn(2, 1, 8, requires_grad=True)
    target = torch.full((2, 1, 8), float("nan"))
    target[:, :, :4] = torch.randn(2, 1, 4)

    loss = MaskedMSE()(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all(), "a NaN gradient would poison every weight"
    # No gradient should flow to the positions the loss never looked at.
    assert torch.equal(prediction.grad[:, :, 4:], torch.zeros(2, 1, 4))


def test_masked_mse_rejects_a_target_with_nothing_to_reconstruct() -> None:
    prediction = torch.zeros(2, 1, 4)
    target = torch.full((2, 1, 4), float("nan"))
    with pytest.raises(ValueError, match="no masked positions"):
        MaskedMSE()(prediction, target)


def test_masked_mse_is_registered() -> None:
    assert "masked_mse" in LOSSES.names()


def test_mae_trains_the_backbone_through_the_generic_step() -> None:
    """MAE's whole training step is DsioModule's shared ``(prediction, target)`` chain --
    no subclass, no ``step()`` override. Moved from the deleted ``tests/ssl/test_methods.
    py``, whose masking/sentinel property already lives in ``test_data.py`` and whose
    mask-aware-loss property is proven above; what is left to check is that the pieces
    compose into a real gradient on a real backbone."""
    torch.manual_seed(0)
    signal = torch.randn(4, CHANNELS, LENGTH)
    hidden = torch.zeros(4, LENGTH, dtype=torch.bool)
    hidden[:, 10:20] = True
    x = signal.masked_fill(hidden.unsqueeze(1), 0.0)
    target = signal.masked_fill(~hidden.unsqueeze(1), float("nan"))

    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    head = mae_decoder_head(DIM, CHANNELS, LENGTH)
    module = DsioModule(backbone=backbone, head=head, loss=MaskedMSE())
    loss = module._common_step({"x": x, "y": target, "row": torch.arange(4)}, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.backbone.parameters())


# --- NT-Xent (SimCLR) and VICReg -----------------------------------------------------
#
# Moved from the deleted tests/ssl/test_methods.py: SimCLR and VICReg were objects with
# their own step(module, x); they are now an ordinary registered (prediction, target) loss
# each, over a batch dsio.nn.data.TwoViewCollate builds. These tests exercise the losses
# directly, the same way test_masked_mse_* above does, rather than through a full module.


@pytest.fixture
def views() -> tuple[torch.Tensor, torch.Tensor]:
    """A batch shaped like TwoViewCollate's output: (x, pair-index target). Built with
    ``nn.Identity`` as the augmentor so both views are literally identical -- irrelevant
    to the loss tests below, which only care about the pair-index contract, not about how
    the two views came to differ."""
    torch.manual_seed(0)
    signal = torch.randn(8, CHANNELS, LENGTH)
    items = [{"x": signal[i], "row": i} for i in range(signal.shape[0])]
    batch = TwoViewCollate(nn.Identity())(items)
    return batch["x"], batch["y"]


def test_nt_xent_produces_a_finite_loss(views: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, target = views
    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    prediction = simclr_projector_head(DIM, out_dim=8)(backbone(x))
    loss = NTXent(temperature=0.2)(prediction, target)
    assert torch.isfinite(loss)


def test_nt_xent_reports_collapse_that_the_loss_hides(
    views: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A collapsed encoder maps everything to one point. Mean cosine goes to 1; the loss
    does not obviously misbehave, and the features are useless."""
    x, target = views
    prediction = torch.ones(x.shape[0], 8)
    diagnostics = NTXent(temperature=0.2).diagnostics(prediction, target, x)
    assert diagnostics["mean_abs_cosine"] == pytest.approx(1.0, abs=1e-3)


def test_nt_xent_needs_negatives() -> None:
    prediction = torch.randn(2, 8)  # one raw window, two views: no negatives at all
    target = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="no negatives"):
        NTXent()(prediction, target)


def test_nt_xent_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        NTXent(temperature=0.0)


def test_nt_xent_loss_falls_when_views_agree() -> None:
    """Sanity on the objective itself: identical views of distinct windows is the easy
    case, over the *same* backbone and head weights as the harder, jittered one."""
    torch.manual_seed(0)
    signal = torch.randn(8, CHANNELS, LENGTH)
    items = [{"x": signal[i], "row": i} for i in range(8)]
    easy_batch = TwoViewCollate(nn.Identity())(items)
    hard_batch = TwoViewCollate(Jitter(3.0))(items)

    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    head = simclr_projector_head(DIM, out_dim=8)
    loss_fn = NTXent(temperature=0.2)

    easy = loss_fn(head(backbone(easy_batch["x"])), easy_batch["y"])
    hard = loss_fn(head(backbone(hard_batch["x"])), hard_batch["y"])
    assert easy.item() < hard.item()


def test_vicreg_produces_a_finite_loss_and_its_terms(
    views: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, target = views
    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    prediction = vicreg_projector_head(DIM, out_dim=8)(backbone(x))
    loss_fn = VICReg()
    loss = loss_fn(prediction, target)
    assert torch.isfinite(loss)
    diagnostics = loss_fn.diagnostics(prediction, target, x)
    assert {"invariance", "variance_penalty", "covariance", "embedding_std"} == set(diagnostics)


def test_vicreg_penalises_a_collapsed_embedding(
    views: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The variance term is what prevents collapse, so it must actually fire on one."""
    x, target = views
    prediction = torch.ones(x.shape[0], 8)
    diagnostics = VICReg(target_std=1.0).diagnostics(prediction, target, x)
    # 0.99 rather than 1.0: the term is relu(target - sqrt(var + 1e-4)), and sqrt(1e-4) is
    # 0.01 -- the epsilon that keeps the gradient finite at exactly zero variance.
    assert diagnostics["variance_penalty"] == pytest.approx(0.99, abs=1e-3)
    assert diagnostics["embedding_std"] == pytest.approx(0.0, abs=1e-4)


def test_vicreg_needs_more_than_one_pair() -> None:
    prediction = torch.randn(2, 8)
    target = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="at least two pairs"):
        VICReg()(prediction, target)


@pytest.mark.parametrize(
    ("head_name", "loss_fn"),
    [("simclr_projector", NTXent(temperature=0.2)), ("vicreg_projector", VICReg())],
    ids=["simclr", "vicreg"],
)
def test_contrastive_losses_train_the_same_module_as_everything_else(
    head_name: str, loss_fn: nn.Module, views: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The property this task exists to prove: DsioModule's one generic step trains a
    contrastive objective exactly the way it trains everything else -- no
    ContrastiveModule, no step() override, just a registered head and a registered loss."""
    x, target = views
    backbone = Conv1dEncoder(channels=CHANNELS, hidden=8, out_dim=DIM, depth=1)
    module = DsioModule(backbone=backbone, head=HEADS.get(head_name)(DIM, out_dim=8), loss=loss_fn)
    loss = module._common_step({"x": x, "y": target, "row": torch.arange(x.shape[0])}, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.backbone.parameters())


# --- contrastive heads ----------------------------------------------------------------


def test_mae_decoder_head_reconstructs_the_original_shape() -> None:
    head = mae_decoder_head(DIM, channels=CHANNELS, length=LENGTH)
    assert head(torch.randn(4, DIM)).shape == (4, CHANNELS, LENGTH)


def test_simclr_projector_head_projects_to_the_configured_dimension() -> None:
    assert simclr_projector_head(DIM, out_dim=8)(torch.randn(4, DIM)).shape == (4, 8)


def test_vicreg_projector_head_projects_to_the_configured_dimension() -> None:
    assert vicreg_projector_head(DIM, out_dim=8)(torch.randn(4, DIM)).shape == (4, 8)


def test_contrastive_heads_and_losses_are_registered() -> None:
    assert {"mae_decoder", "simclr_projector", "vicreg_projector"} <= set(HEADS.names())
    assert {"masked_mse", "nt_xent", "vicreg"} <= set(LOSSES.names())


def test_augmentors_are_registered() -> None:
    assert {"jitter", "random_scale", "none"} <= set(AUGMENTORS.names())
