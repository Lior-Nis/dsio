"""Identity-preserving raw windows and deterministic DataLoader construction."""

from __future__ import annotations

from typing import Any, NewType, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dsio.batches import (
    BatchLoader,
    TrainingBatch,
    TrainingItem,
    WindowBatch,
    WindowItem,
)
from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex, assert_index_matches_store
from dsio.runs.seeding import dataloader_kwargs

_TargetDataset = NewType("_TargetDataset", Dataset[TrainingItem])


class WindowDataset(Dataset[WindowItem]):
    """Windows of one store, restricted to a set of index positions.

    ``positions`` are offsets into ``index``, which is what a
    :class:`~dsio.data.splits.folds.Fold` carries. Each item reports the position it came from
    so predictions can be realigned by identity rather than by trusting loader ordering.

    ``payload_dtype`` is what lets a token-id corpus reach a model as ids rather than as
    numbers. Left ``None`` — the default — the window is cast to float32 exactly as it
    always was, whatever the store holds; set to an integer dtype it is cast to that
    instead, which is the whole difference between ``nn.Embedding`` working and refusing
    the batch outright. The unconditional ``.float()`` this used to do was worse than it
    sounds: ``nn.Embedding`` at least fails loudly on float indices, while the ``conv1d``
    backbone happily convolves token ids into plausible-looking numbers and trains.

    **It is opted into, not inferred from the store's dtype.** Inference would be more
    convenient and it is the wrong call, because the dtype on disk answers "how are these
    bytes packed", not "are these numbers or symbols". :class:`~dsio.data.format.DTypeCode`
    carries ``INT16``/``INT8`` precisely so a corpus can be stored as quantised ADC
    counts — those are signal, they want the float path, and inference would start handing
    them to a model as integers, where ``InstanceStandardize`` does integer division and a
    convolution trains on something subtly wrong. The same int32 bytes are token ids in one
    store and sensor counts in another; only the caller knows which, so the caller says so.

    Stochastic training augmentation deliberately does not live here. Every stage receives
    the same raw, identity-bearing sample; the Lightning module owns the train-only lane.
    """

    def __init__(
        self,
        store: SignalStore,
        index: WindowIndex,
        positions: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        *,
        channels_first: bool = True,
        payload_dtype: torch.dtype | None = None,
    ) -> None:
        assert_index_matches_store(store, index)
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
        self.payload_dtype = payload_dtype

    def __len__(self) -> int:
        return int(self.positions.size)

    def __getitem__(self, i: int) -> WindowItem:
        position = int(self.positions[i])
        start = int(self.index.starts[position])
        window = self.store.read(start, self.index.spec.length)
        # The store is [time, channels]; torch convolutions want [channels, time]. The copy
        # is required because the mmap slice is a view, and a view handed to a worker
        # process outlives the read it came from.
        array = np.ascontiguousarray(window.T if self.channels_first else window)
        # float32 unless the caller asked for something else: a signal payload is float
        # regardless of how its bytes are packed, and a token id is not a number at all.
        raw = torch.from_numpy(array)
        x = raw.float() if self.payload_dtype is None else raw.to(self.payload_dtype)
        # Predictions are aligned by identity, not by trusting loader order.
        item = WindowItem(
            sample_id=self.index.sample_id(position),
            x=x,
            row=position,
        )
        if self.labels is not None:
            item["y"] = torch.as_tensor(self.labels[position])
        return item

    @property
    def groups(self) -> np.ndarray:
        """Group per item, for verifying a loader never mixes a split boundary."""
        return self.index.groups[self.positions]


def train_dataset(
    store: SignalStore,
    index: WindowIndex,
    positions: np.ndarray | None = None,
    *,
    labels: np.ndarray | None = None,
    channels_first: bool = True,
    payload_dtype: torch.dtype | None = None,
) -> WindowDataset:
    """Build raw training windows; train-only stochastic work belongs to DsioModule."""
    return WindowDataset(
        store,
        index,
        positions,
        labels=labels,
        channels_first=channels_first,
        payload_dtype=payload_dtype,
    )


def val_dataset(
    store: SignalStore,
    index: WindowIndex,
    positions: np.ndarray | None = None,
    *,
    labels: np.ndarray | None = None,
    channels_first: bool = True,
    payload_dtype: torch.dtype | None = None,
) -> WindowDataset:
    """Build raw evaluation windows."""
    return WindowDataset(
        store,
        index,
        positions,
        labels=labels,
        channels_first=channels_first,
        payload_dtype=payload_dtype,
    )


def labelled_dataset(
    store: SignalStore,
    index: WindowIndex,
    positions: np.ndarray | None = None,
    *,
    labels: np.ndarray,
    channels_first: bool = True,
    payload_dtype: torch.dtype | None = None,
) -> _TargetDataset:
    """Build windows whose label is statically guaranteed to be present."""
    return _TargetDataset(
        cast(
            "Dataset[TrainingItem]",
            val_dataset(
                store,
                index,
                positions,
                labels=labels,
                channels_first=channels_first,
                payload_dtype=payload_dtype,
            ),
        )
    )


def make_loader(
    dataset: Dataset[WindowItem],
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> BatchLoader[WindowBatch]:
    """Build a DataLoader whose *shuffle order* does not depend on the worker count.

    Seeding the process is not enough: each worker gets its own RNG, so without an explicit
    generator and ``worker_init_fn`` the shuffle order varies with ``num_workers`` — which
    would make a result depend on a performance knob and put it straight into dsio's list
    of things that must never change an answer. ``dataloader_kwargs`` closes that.

    Stochastic augmentation is absent from this layer, so worker scheduling can affect
    throughput but not the random stream used to build training views.
    """
    kwargs: dict[str, Any] = dict(dataloader_kwargs(seed))
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return cast(
        "BatchLoader[WindowBatch]",
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            **kwargs,
        ),
    )


def make_target_loader(
    dataset: _TargetDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> BatchLoader[TrainingBatch]:
    """Build a loader that statically preserves a target-bearing dataset contract."""
    return cast(
        "BatchLoader[TrainingBatch]",
        make_loader(
            cast("Dataset[WindowItem]", dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            seed=seed,
            drop_last=drop_last,
        ),
    )
