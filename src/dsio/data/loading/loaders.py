"""Deterministic native PyTorch DataLoader construction."""

from __future__ import annotations

import pickle
import random
from collections.abc import Mapping, Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler

from dsio.data.loading.collation import Collate, IdentityCollator
from dsio.data.loading.datasets import LoadingError

_MAX_SEED = 2**32 - 1


def build_loader[Item: Mapping[str, Any]](
    dataset: Dataset[Item],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    collate_fn: Collate | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build one seeded loader after validating arguments and worker safety."""
    validate_loader_options(batch_size=batch_size, num_workers=num_workers, seed=seed)
    if not isinstance(shuffle, bool):
        raise LoadingError(f"shuffle must be bool, got {type(shuffle).__name__}")
    if not isinstance(drop_last, bool):
        raise LoadingError(f"drop_last must be bool, got {type(drop_last).__name__}")
    if collate_fn is not None and not callable(collate_fn):
        raise LoadingError("collate_fn must be callable")
    collator = IdentityCollator(collate_fn)
    if num_workers:
        _require_picklable(dataset, "dataset")
        _require_picklable(collator, "collator")

    kwargs: dict[str, Any] = dict(dataloader_kwargs(seed))
    if num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    sampler = None
    if shuffle:
        sampler_generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(cast("Sized", dataset), generator=sampler_generator)
    return DataLoader(
        cast("Dataset[dict[str, Any]]", dataset),
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collator,
        **kwargs,
    )


def validate_loader_options(*, batch_size: object, num_workers: object, seed: object) -> None:
    """Validate shared loader scalars before a DataModule reaches setup."""
    _positive_integer("batch_size", batch_size)
    _nonnegative_integer("num_workers", num_workers)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise LoadingError(f"seed must be an integer in [0, {_MAX_SEED}], got {seed!r}")


def dataloader_kwargs(seed: int) -> dict[str, object]:
    """Return deterministic generator and worker seeding arguments for a DataLoader."""
    generator = torch.Generator().manual_seed(seed)
    return {"generator": generator, "worker_init_fn": _seed_worker}


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _require_picklable(value: object, name: str) -> None:
    try:
        pickle.dumps(value)
    except Exception as error:
        raise LoadingError(
            f"{name} cannot be used with DataLoader workers because it is not picklable: {error}"
        ) from error


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoadingError(f"{name} must be a positive integer, got {value!r}")


def _nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LoadingError(f"{name} must be a non-negative integer, got {value!r}")
