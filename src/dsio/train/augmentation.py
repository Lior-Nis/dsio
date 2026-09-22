"""Stateless, accelerator-side augmentation of collated training batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from dsio.contracts import sha256_of
from dsio.model.masking import apply_mask

type Batch = Mapping[str, Any]
type Mask = Callable[[Tensor, torch.Generator | None], Tensor]


class AugmentationError(ValueError):
    """A training batch or augmentation violated the accelerator-side contract."""


class MaskedReconstruction(nn.Module):
    """Create a masked input and NaN-sentinel reconstruction target on the input device."""

    def __init__(self, mask: Mask, *, normalize_target: bool = True) -> None:
        super().__init__()
        if not callable(mask):
            raise AugmentationError("mask must be callable")
        self.mask = mask
        self.normalize_target = normalize_target

    def forward(
        self,
        batch: Batch,
        *,
        seed: int,
        epoch: int,
        step: int,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        x, sample_ids = _inputs(batch)
        generator = _generator(x, seed, epoch, step, sample_ids, identity, "masked")
        hidden = self.mask(x, generator)
        if hidden.dtype != torch.bool or hidden.shape != (x.shape[0], x.shape[-1]):
            raise AugmentationError(
                "mask must return a boolean [batch, time] tensor on the input device"
            )
        if hidden.device != x.device:
            raise AugmentationError("mask must remain on the input device")
        target = x
        if self.normalize_target:
            target = (x - x.mean(dim=-1, keepdim=True)) / (
                x.std(dim=-1, keepdim=True) + 1e-6
            )
        return {
            **batch,
            "x": apply_mask(x, hidden),
            "y": apply_mask(target, ~hidden, value=float("nan")),
        }


class TwoView(nn.Module):
    """Create two reproducible augmented views and their symmetric pair indices."""

    def __init__(
        self,
        augmentor: nn.Module,
        *,
        views: tuple[str, str] = ("view-0", "view-1"),
    ) -> None:
        super().__init__()
        if not isinstance(augmentor, nn.Module):
            raise AugmentationError("augmentor must be a torch nn.Module")
        if len(set(views)) != 2 or any(not view for view in views):
            raise AugmentationError("two-view identities must be distinct non-empty strings")
        self.augmentor = augmentor
        self.views = views

    def forward(
        self,
        batch: Batch,
        *,
        seed: int,
        epoch: int,
        step: int,
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        x, sample_ids = _inputs(batch)
        generated: list[Tensor] = []
        with torch.no_grad():
            for view in self.views:
                generator = _generator(x, seed, epoch, step, sample_ids, identity, view)
                value = self.augmentor(x, generator=generator)
                if not isinstance(value, Tensor) or value.shape != x.shape:
                    shape = None if not isinstance(value, Tensor) else tuple(value.shape)
                    raise AugmentationError(
                        f"augmentor must preserve input shape {tuple(x.shape)}, got {shape}"
                    )
                if value.device != x.device:
                    raise AugmentationError("augmentor must remain on the input device")
                generated.append(value)
        size = x.shape[0]
        result = dict(batch)
        result.update(
            sample_id=sample_ids * 2,
            x=torch.cat(generated, dim=0),
            y=(torch.arange(2 * size, device=x.device) + size) % (2 * size),
            view_id=[self.views[0]] * size + [self.views[1]] * size,
        )
        row = batch.get("row")
        if isinstance(row, Tensor):
            result["row"] = row.repeat(2)
        return result


def _inputs(batch: Batch) -> tuple[Tensor, list[str]]:
    x = batch.get("x")
    if not isinstance(x, Tensor) or x.ndim < 1:
        raise AugmentationError("training augmentation requires tensor batch['x']")
    identities = batch.get("sample_id")
    if isinstance(identities, str) or not isinstance(identities, Sequence):
        raise AugmentationError("training augmentation requires sample_id strings")
    sample_ids = list(identities)
    if (
        len(sample_ids) != x.shape[0]
        or not sample_ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise AugmentationError(
            "training augmentation requires one non-empty sample_id string per input"
        )
    return x, cast("list[str]", sample_ids)


def _generator(
    x: Tensor,
    seed: int,
    epoch: int,
    step: int,
    sample_ids: list[str],
    identity: Mapping[str, Any],
    view: str,
) -> torch.Generator:
    digest = sha256_of(
        {
            "version": 1,
            "seed": seed,
            "epoch": epoch,
            "step": step,
            "sample_ids": sample_ids,
            "augmentation": identity,
            "view": view,
        }
    )
    derived = int(digest[:16], 16) & ((1 << 63) - 1)
    return torch.Generator(device=x.device).manual_seed(derived)


__all__ = ["AugmentationError", "MaskedReconstruction", "TwoView"]
