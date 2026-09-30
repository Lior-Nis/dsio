"""Dataset boundaries that keep governed sample identity attached to payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Sized
from typing import Any

from torch.utils.data import Dataset, IterableDataset

from dsio.data.examples import Examples
from dsio.data.store import SignalStore


class LoadingError(ValueError):
    """A dataset or loader contradicted its governed data contract."""


type DataItem = Mapping[str, Any]
type DatasetFactory = Callable[
    [SignalStore, Examples, Sequence[str]],
    Dataset[DataItem],
]


class IdentityDataset(Dataset[dict[str, Any]]):
    """Require an injected dataset to expose the exact assigned identity at each position."""

    def __init__(self, dataset: Dataset[DataItem], sample_ids: Sequence[str]) -> None:
        self.dataset = dataset
        self.sample_ids = tuple(sample_ids)
        if isinstance(dataset, IterableDataset):
            raise LoadingError(
                "dataset factory must return a map-style Dataset, not IterableDataset"
            )
        if not isinstance(dataset, Sized):
            raise LoadingError("dataset factory returned a dataset without a length")
        try:
            size = len(dataset)
        except Exception as error:
            raise LoadingError(f"dataset length could not be read: {error}") from error
        if size != len(self.sample_ids):
            raise LoadingError(
                f"dataset factory returned {size} items but the role assignment has "
                f"{len(self.sample_ids)} sample identities"
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> dict[str, Any]:
        try:
            item = self.dataset[position]
        except Exception as error:
            raise LoadingError(
                f"dataset could not read assigned sample {self.sample_ids[position]!r}: {error}"
            ) from error
        if not isinstance(item, Mapping):
            raise LoadingError(
                f"dataset item {position} must be a mapping containing sample_id, got "
                f"{type(item).__name__}"
            )
        expected = self.sample_ids[position]
        actual = item.get("sample_id")
        if not isinstance(actual, str):
            raise LoadingError(
                f"dataset item {position} must contain string sample_id {expected!r}"
            )
        if actual != expected:
            raise LoadingError(
                f"dataset item {position} changed sample_id from {expected!r} to {actual!r}"
            )
        return dict(item)
