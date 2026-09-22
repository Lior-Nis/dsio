"""Identity-preserving collation for every CPU-side data path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import default_collate

from dsio.data.loading.datasets import DataItem, LoadingError

Batch = dict[str, Any]
Collate = Callable[[list[DataItem]], Mapping[str, Any]]
_MAX_BATCH_DEPTH = 64


def collate_items(
    items: list[DataItem],
    *,
    collate_fn: Collate | None = None,
) -> Batch:
    """Collate mappings while requiring exact ordered sample identity in the result."""
    if not items:
        raise LoadingError("cannot collate an empty batch")
    expected = _item_ids(items)
    try:
        batch = default_collate(items) if collate_fn is None else collate_fn(items)
    except LoadingError:
        raise
    except Exception as error:
        raise LoadingError(f"batch collation failed: {error}") from error
    if not isinstance(batch, Mapping):
        raise LoadingError(
            f"collation must return a mapping containing sample_id, got {type(batch).__name__}"
        )
    _validate_batch(batch, "batch", len(expected), set(), 0)
    actual = _batch_ids(batch.get("sample_id"))
    if actual != expected:
        raise LoadingError(
            f"collation changed sample_id order or membership: expected {expected}, got {actual}"
        )
    result = dict(batch)
    result["sample_id"] = expected
    return result


class IdentityCollator:
    """Pickle-friendly adapter that applies a configured collator behind the identity guard."""

    def __init__(self, collate_fn: Collate | None = None) -> None:
        self.collate_fn = collate_fn

    def __call__(self, items: list[DataItem]) -> Batch:
        return collate_items(items, collate_fn=self.collate_fn)


def _item_ids(items: list[DataItem]) -> list[str]:
    identities: list[str] = []
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise LoadingError(
                f"batch item {position} must be a mapping containing sample_id, got "
                f"{type(item).__name__}"
            )
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str):
            raise LoadingError(f"batch item {position} has no string sample_id")
        identities.append(sample_id)
    return identities


def _batch_ids(value: object) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise LoadingError("collated batch must contain sample_id as a sequence of strings")
    identities = list(value)
    if any(not isinstance(sample_id, str) for sample_id in identities):
        raise LoadingError("collated batch sample_id values must all be strings")
    return identities


def _validate_batch(
    value: object,
    path: str,
    batch_size: int,
    active: set[int],
    depth: int,
) -> None:
    if depth > _MAX_BATCH_DEPTH:
        raise LoadingError(f"{path} exceeds the maximum batch nesting depth")
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise LoadingError(
                f"collation must keep tensors on CPU; {path} is on {value.device.type!r}"
            )
        _require_cardinality(value.ndim, value.shape[0] if value.ndim else None, path, batch_size)
        return
    if isinstance(value, np.ndarray):
        _require_cardinality(value.ndim, value.shape[0] if value.ndim else None, path, batch_size)
        return
    if isinstance(value, set | frozenset):
        raise LoadingError(f"{path} uses an unordered batch container")
    if isinstance(value, Mapping):
        _enter_container(value, path, active)
        try:
            for key, item in value.items():
                _validate_batch(item, f"{path}.{key}", batch_size, active, depth + 1)
        finally:
            active.remove(id(value))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        _enter_container(value, path, active)
        try:
            if all(
                item is None or isinstance(item, str | bytes | bool | int | float) for item in value
            ):
                _require_cardinality(1, len(value), path, batch_size)
            for index, item in enumerate(value):
                _validate_batch(item, f"{path}[{index}]", batch_size, active, depth + 1)
        finally:
            active.remove(id(value))


def _require_cardinality(
    dimensions: int,
    size: int | None,
    path: str,
    batch_size: int,
) -> None:
    if dimensions and size != batch_size:
        raise LoadingError(
            f"collated field {path} has {size} rows for {batch_size} sample_id values"
        )


def _enter_container(value: object, path: str, active: set[int]) -> None:
    marker = id(value)
    if marker in active:
        raise LoadingError(f"{path} contains a recursive batch container")
    active.add(marker)
