"""Masking strategies, as their own module.

Masking is separated from the methods that use it because it is the axis that actually
gets varied: several "different" pretraining setups are the same masked autoencoder with a
different masking mode. Making masking a slot rather than a constructor argument means a new
mode is a new strategy class, not a new pipeline.

Every strategy returns a boolean ``hidden`` tensor that is ``True`` where a position is
**hidden**. That convention is stated once here and asserted in the tests, because the
opposite convention is equally natural and mixing them produces a model that trains on exactly
the positions it was supposed to predict — a leak that shows up as a suspiciously good
reconstruction loss and nothing else. DSio batch ``mask`` fields mean the opposite (``True``
marks a valid, scoreable position), so a hidden tensor is never called ``mask``.

Masks are generated per-sample, never once per batch. A batch-wide mask correlates what
every sample in the batch has to infer, which quietly reduces the effective difficulty of
the task and makes the batch size a hyperparameter of the objective.

Legacy experimental (docs/component-admission.md): no real downstream use yet, no
compatibility promise, deleted at 1.0 if still unproven.
"""

from __future__ import annotations

import torch


def _check(ratio: float) -> float:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must be in (0, 1), got {ratio}")
    return ratio


class RandomMask:
    """Mask individual timesteps uniformly at random.

    The weakest of the strategies on continuous signal: neighbouring samples are highly
    correlated, so a randomly hidden timestep is recoverable by interpolation and the model
    learns a smoother rather than a representation. Kept because it is the honest baseline
    the structured strategies must beat.
    """

    def __init__(self, ratio: float = 0.5) -> None:
        self.ratio = _check(ratio)

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        batch, _, length = x.shape
        noise = torch.rand(batch, length, device=x.device, generator=generator)
        keep = round(length * (1.0 - self.ratio))
        order = noise.argsort(dim=-1)
        hidden = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        hidden.scatter_(-1, order[:, :keep], False)
        return hidden


class SpanMask:
    """Mask contiguous runs, the SpanBERT / wav2vec 2.0 shape.

    Forces the model to infer from context rather than interpolate between neighbours,
    which is what makes the task non-trivial on an oversampled signal. The span count is
    derived from the target ratio so the two knobs cannot disagree.
    """

    def __init__(self, ratio: float = 0.5, span: int = 20) -> None:
        self.ratio = _check(ratio)
        if span < 1:
            raise ValueError(f"span must be at least 1, got {span}")
        self.span = span

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        batch, _, length = x.shape
        span = min(self.span, length)
        n_spans = max(1, round(length * self.ratio / span))
        hidden = torch.zeros(batch, length, dtype=torch.bool, device=x.device)
        starts = torch.randint(
            0, max(1, length - span + 1), (batch, n_spans), device=x.device, generator=generator
        )
        offsets = torch.arange(span, device=x.device)
        # Spans may overlap. That is deliberate: rejecting overlaps would bias starts away
        # from each other and make the masked positions less clustered than configured.
        positions = (starts.unsqueeze(-1) + offsets).clamp_(max=length - 1)
        hidden.scatter_(1, positions.reshape(batch, -1), True)
        return hidden


class PatchMask:
    """Mask whole non-overlapping patches, the MAE shape.

    The right strategy when the backbone tokenises into patches anyway: masking at a
    different granularity than the model's own tokens means a token is partially visible,
    and a partially visible token leaks its own target.
    """

    def __init__(self, ratio: float = 0.75, patch: int = 16) -> None:
        self.ratio = _check(ratio)
        if patch < 1:
            raise ValueError(f"patch must be at least 1, got {patch}")
        self.patch = patch

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        batch, _, length = x.shape
        patch = min(self.patch, length)
        n_patches = length // patch
        if n_patches == 0:  # pragma: no cover - guarded by the clamp above
            raise ValueError(f"window of {length} is shorter than one patch of {patch}")

        noise = torch.rand(batch, n_patches, device=x.device, generator=generator)
        n_masked = max(1, round(n_patches * self.ratio))
        order = noise.argsort(dim=-1)
        patch_hidden = torch.zeros(batch, n_patches, dtype=torch.bool, device=x.device)
        patch_hidden.scatter_(-1, order[:, :n_masked], True)

        hidden = patch_hidden.repeat_interleave(patch, dim=-1)
        if hidden.shape[-1] < length:
            # The tail that does not fill a patch stays visible rather than being masked
            # as a short patch, which would make it an easier target than every other.
            pad = torch.zeros(batch, length - hidden.shape[-1], dtype=torch.bool, device=x.device)
            hidden = torch.cat([hidden, pad], dim=-1)
        return hidden


class CausalMask:
    """Hide the final fraction of the window: forecasting as a pretext task.

    Deterministic, so it is the one strategy whose mask does not depend on the generator.
    It is also the only one whose pretext matches a real downstream task, which makes it
    the natural pretraining objective when the deployment question is "what happens next".
    """

    def __init__(self, ratio: float = 0.25) -> None:
        self.ratio = _check(ratio)

    def __call__(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        batch, _, length = x.shape
        cut = length - max(1, round(length * self.ratio))
        hidden = torch.zeros(batch, length, dtype=torch.bool, device=x.device)
        hidden[:, cut:] = True
        return hidden


def apply_mask(x: torch.Tensor, hidden: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Return ``x`` with hidden positions replaced, broadcasting ``hidden`` over channels.

    Never in place. The unmasked original is the reconstruction target, and overwriting it
    would make the target equal to the input — a loss that goes to zero while the model
    learns nothing, and which looks like spectacular convergence.
    """
    return x.masked_fill(hidden.unsqueeze(1), value)


__all__ = [
    "CausalMask",
    "PatchMask",
    "RandomMask",
    "SpanMask",
    "apply_mask",
]
