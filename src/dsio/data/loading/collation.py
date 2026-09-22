"""Identity-preserving collation for every CPU-side data path."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import Tensor
from torch.utils.data import default_collate

from dsio.data.loading.datasets import DataItem, LoadingError

Batch = dict[str, Any]
Collate = Callable[[list[DataItem]], Mapping[str, Any]]


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
    _require_cpu(batch, "batch")
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


def _require_cpu(value: object, path: str) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise LoadingError(
                f"collation must keep tensors on CPU; {path} is on {value.device.type!r}"
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_cpu(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _require_cpu(item, f"{path}[{index}]")
