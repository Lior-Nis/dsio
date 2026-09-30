"""Identity-preserving windows over the canonical memory-mapped store."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dsio.batches import WindowItem
from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex, assert_index_matches_store


class WindowDataset(Dataset[WindowItem]):
    """Read selected index positions as raw, identity-bearing tensor windows.

    ``positions`` are offsets into ``index``. Each item reports its governed sample ID and
    original row position so batching never destroys identity.

    ``payload_dtype`` is explicit because storage dtype does not determine semantics. An
    int32 store may contain token IDs, which require an integer tensor, or quantised sensor
    values, which require the default float32 tensor. Only the caller knows which.

    Stochastic augmentation does not live here. Every phase receives the same raw sample;
    the Lightning module owns the train-only accelerator lane.
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

    def __getitem__(self, item: int) -> WindowItem:
        position = int(self.positions[item])
        start = int(self.index.starts[position])
        window = self.store.read(start, self.index.spec.length)
        array = np.ascontiguousarray(window.T if self.channels_first else window)
        raw = torch.from_numpy(array)
        payload = raw.float() if self.payload_dtype is None else raw.to(self.payload_dtype)
        result = WindowItem(
            sample_id=self.index.sample_id(position),
            x=payload,
            row=position,
        )
        if self.labels is not None:
            result["y"] = torch.as_tensor(self.labels[position])
        return result

    @property
    def groups(self) -> np.ndarray:
        """Group per item, for verifying a loader never crosses a split boundary."""
        return self.index.groups[self.positions]
