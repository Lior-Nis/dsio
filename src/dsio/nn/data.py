"""A Dataset over the memory-mapped store, built from fold positions.

This is where Phase 2 pays for itself. A fold is integer positions into a
:class:`~dsio.data.views.WindowIndex`; a loader over it reads windows from the mmap on
demand. Nothing is copied, so five folds over 42M windows cost five position arrays rather
than five datasets — and switching window length costs a new index, not a new corpus.

The store's reader is keyed by process id and its live handles are dropped on pickling, so
worker processes reopen their own. That was verified under real DataLoader concurrency in
ADR 0005's worker-scaling benchmark before anything depended on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.runs.seeding import dataloader_kwargs


class WindowDataset(Dataset[dict[str, Any]]):
    """Windows of one store, restricted to a set of index positions.

    ``positions`` are offsets into ``index``, which is what a
    :class:`~dsio.eval.contract.Fold` carries. Each item reports the position it came from
    so predictions can be realigned by identity rather than by trusting loader ordering.
    """

    def __init__(
        self,
        store: SignalStore,
        index: WindowIndex,
        positions: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        *,
        channels_first: bool = True,
    ) -> None:
        if index.store_name != store.path.name:
            raise ValueError(
                f"index was built for store {index.store_name!r}, not {store.path.name!r}"
            )
        self.store = store
        self.index = index
        self.positions = (
            np.arange(len(index), dtype=np.int64)
            if positions is None
            else np.asarray(positions, dtype=np.int64)
        )
        source = index.labels if labels is None else labels
        self.labels = None if source is None else np.asarray(source)
        if self.labels is not None and len(self.labels) != len(index):
            raise ValueError(
                f"labels has {len(self.labels)} entries for {len(index)} windows; they must "
                "be aligned with the whole index, not with this fold"
            )
        self.channels_first = channels_first

    def __len__(self) -> int:
        return int(self.positions.size)

    def __getitem__(self, i: int) -> dict[str, Any]:
        position = int(self.positions[i])
        start = int(self.index.starts[position])
        window = self.store.read(start, self.index.spec.length)
        # The store is [time, channels]; torch convolutions want [channels, time]. The copy
        # is required because the mmap slice is a view, and a view handed to a worker
        # process outlives the read it came from.
        array = np.ascontiguousarray(window.T if self.channels_first else window)
        item: dict[str, Any] = {
            "x": torch.from_numpy(array).float(),
            "row": position,
        }
        if self.labels is not None:
            value = self.labels[position]
            item["y"] = torch.as_tensor(value)
        return item

    @property
    def groups(self) -> np.ndarray:
        """Group per item, for verifying a loader never mixes a split boundary."""
        return self.index.groups[self.positions]


def make_loader(
    dataset: WindowDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Build a DataLoader whose result does not depend on the worker count.

    Seeding the process is not enough: each worker gets its own RNG, so without an explicit
    generator and ``worker_init_fn`` the shuffle order and any augmentation randomness vary
    with ``num_workers`` — which would make a result depend on a performance knob and put
    it straight into dsio's list of things that must never change an answer.
    """
    kwargs: dict[str, Any] = dict(dataloader_kwargs(seed))
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        **kwargs,
    )
