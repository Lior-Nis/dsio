"""Masking invariants. The convention test is the one that matters most."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dsio.ssl.masking import (  # noqa: E402
    MASKS,
    CausalMask,
    PatchMask,
    RandomMask,
    SpanMask,
    apply_mask,
)


@pytest.fixture
def signal() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 3, 256)


ALL = [RandomMask(0.5), SpanMask(0.5, span=16), PatchMask(0.5, patch=16), CausalMask(0.5)]


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: type(s).__name__)
def test_true_means_hidden(strategy, signal: torch.Tensor) -> None:
    """The convention, asserted rather than documented.

    The opposite convention is equally natural, and mixing them trains the model on exactly
    the positions it was meant to predict — a leak whose only symptom is a suspiciously
    good reconstruction loss.
    """
    mask = strategy(signal)
    masked = apply_mask(signal, mask, value=0.0)
    assert (masked[:, :, mask[0]][0] == 0).all() or not mask[0].any()
    kept = ~mask
    assert torch.equal(masked[0][:, kept[0]], signal[0][:, kept[0]])


@pytest.mark.parametrize("strategy", ALL, ids=lambda s: type(s).__name__)
def test_mask_shape_is_per_sample_over_time(strategy, signal: torch.Tensor) -> None:
    mask = strategy(signal)
    assert mask.shape == (8, 256)
    assert mask.dtype == torch.bool


@pytest.mark.parametrize("strategy", ALL[:3], ids=lambda s: type(s).__name__)
def test_masks_differ_between_samples(strategy, signal: torch.Tensor) -> None:
    """A batch-wide mask correlates what every sample must infer, which quietly lowers the
    difficulty and makes batch size a hyperparameter of the objective."""
    mask = strategy(signal)
    assert not torch.equal(mask[0], mask[1])


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [(RandomMask(0.5), 0.5), (PatchMask(0.75, patch=16), 0.75), (CausalMask(0.25), 0.25)],
)
def test_ratio_is_approximately_honoured(strategy, expected: float, signal: torch.Tensor) -> None:
    assert strategy(signal).float().mean().item() == pytest.approx(expected, abs=0.05)


def test_span_mask_produces_contiguous_runs(signal: torch.Tensor) -> None:
    """Contiguity is the point: a randomly hidden timestep on an oversampled signal is
    recoverable by interpolation, so the model learns a smoother rather than a
    representation."""
    mask = SpanMask(0.3, span=32)(signal)
    row = mask[0]
    transitions = (row[1:] != row[:-1]).sum().item()
    assert transitions <= 2 * (256 // 32), "too many boundaries to be spans"
    assert row.any()


def test_patch_mask_hides_whole_patches(signal: torch.Tensor) -> None:
    """Masking at a finer granularity than the model's tokens leaks each token's target."""
    mask = PatchMask(0.5, patch=32)(signal)
    for patch in mask[0].split(32):
        assert patch.all() or not patch.any()


def test_causal_mask_hides_only_the_tail(signal: torch.Tensor) -> None:
    mask = CausalMask(0.25)(signal)
    assert not mask[:, :192].any()
    assert mask[:, 192:].all()


def test_causal_mask_is_deterministic(signal: torch.Tensor) -> None:
    """The one strategy whose mask does not depend on the generator."""
    strategy = CausalMask(0.25)
    assert torch.equal(strategy(signal), strategy(signal))


def test_apply_mask_never_mutates_its_input(signal: torch.Tensor) -> None:
    """The unmasked original is the target; overwriting it makes the loss go to zero while
    the model learns nothing, which looks like spectacular convergence."""
    original = signal.clone()
    apply_mask(signal, RandomMask(0.5)(signal))
    assert torch.equal(signal, original)


def test_apply_mask_broadcasts_over_channels(signal: torch.Tensor) -> None:
    mask = torch.zeros(8, 256, dtype=torch.bool)
    mask[:, :10] = True
    masked = apply_mask(signal, mask)
    assert (masked[:, :, :10] == 0).all()


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_degenerate_ratios_are_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match="mask ratio"):
        RandomMask(ratio)


def test_strategies_are_registered() -> None:
    assert {"random", "span", "patch", "causal"} <= set(MASKS.names())


def test_a_generator_makes_masking_reproducible(signal: torch.Tensor) -> None:
    strategy = RandomMask(0.5)
    first = strategy(signal, generator=torch.Generator().manual_seed(3))
    second = strategy(signal, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
