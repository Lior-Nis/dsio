"""Whole-sample store dataset (legacy experimental).

Legacy experimental under the pre-1.0 clause in docs/component-admission.md: no real
downstream use yet, no compatibility promise, deleted at 1.0 if still unproven. The
warehouse field-mapping dataset (Component Warehouse v1, Story 7.1) reshapes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dsio.data.examples import Examples
from dsio.data.loading.datasets import DataItem, LoadingError
from dsio.data.store import SignalStore, StoreError


class StoredSamples(Dataset[dict[str, Any]]):
    """Decode whole canonical-store samples into identity-bearing tensor items."""

    def __init__(self, store: SignalStore, sample_ids: Sequence[str]) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)
        for sample_id in self.sample_ids:
            try:
                store.entity(sample_id)
            except StoreError as error:
                raise LoadingError(
                    f"stored-sample dataset cannot resolve assigned sample {sample_id!r}: {error}"
                ) from error

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> dict[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        data = np.array(sample["data"], copy=True, order="C")
        return {"sample_id": sample["sample_id"], "x": torch.from_numpy(data)}


def stored_samples(
    store: SignalStore,
    examples: Examples,
    sample_ids: Sequence[str],
) -> Dataset[DataItem]:
    """Whole-sample factory: one item per persisted sample.

    Window factories use ``examples`` to resolve derived identities into view positions;
    this entity-level factory uses it to prove the supplied store is the corpus whose split
    is being replayed.
    """
    digest = store.identity
    if examples.name != store.path.name or examples.digest != digest:
        raise LoadingError(
            f"stored-sample factory received store {store.path.name!r} with content identity "
            f"{digest!r}, but examples describe {examples.name!r} with {examples.digest!r}"
        )
    return StoredSamples(store, sample_ids)
