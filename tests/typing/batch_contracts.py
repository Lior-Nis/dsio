"""Static assertions for the item, loader, model, and prediction boundaries."""

from __future__ import annotations

from typing import Any, assert_type, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsio.batches import (
    BatchLoader,
    LoaderBatch,
    PredictionBatch,
    TrainingBatch,
    WindowItem,
)
from dsio.data.loading import WindowDataset, build_loader
from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.model.module import DsioModule


def check_batch_contracts(
    item: WindowItem,
    loader: BatchLoader[LoaderBatch],
    module: DsioModule,
    training_batch: TrainingBatch,
    predictions: list[PredictionBatch],
    labels: np.ndarray,
    store: SignalStore,
    index: WindowIndex,
) -> None:
    """Keep framework ``Any`` annotations from masking contract regressions."""
    assert_type(item["x"], torch.Tensor)
    assert_type(item["row"], int)
    assert_type(item.get("y"), torch.Tensor | None)
    for batch in loader:
        assert_type(batch, LoaderBatch)
    assert_type(module.predict_step(training_batch, 0), PredictionBatch)
    window_loader = build_loader(WindowDataset(store, index, labels=labels))
    assert_type(window_loader, DataLoader[dict[str, Any]])
    for batch in window_loader:
        assert_type(module.training_step(cast(TrainingBatch, batch), 0), torch.Tensor)
