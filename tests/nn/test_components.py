"""Registered components: backbones, heads, losses, transforms, augmentors.

The mask-aware loss gets the most scrutiny here, because it is the whole point of Task
6a: a NaN sentinel in the target (see ``WindowDataset`` in ``dsio.nn.data``) is only
useful if a loss that reads it naively actually breaks, and a loss that selects on it
before computing error actually rewards reconstruction over copying.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from dsio.nn.components import MaskedMSE  # noqa: E402
from dsio.nn.registry import LOSSES  # noqa: E402


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
