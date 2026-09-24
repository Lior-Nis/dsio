"""Static assertions for the item, loader, model, and prediction boundaries."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import torch

from dsio.batches import (
    BatchLoader,
    LoaderBatch,
    PredictionBatch,
    TrainingBatch,
    WindowBatch,
    WindowItem,
)
from dsio.data.store import SignalStore
from dsio.data.views import WindowIndex
from dsio.dataset.dataset import (
    WindowDataset,
    labelled_dataset,
    make_loader,
    make_target_loader,
)
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
    window_loader = make_loader(WindowDataset(store, index))
    assert_type(window_loader, BatchLoader[WindowBatch])
    labelled_loader = make_target_loader(labelled_dataset(store, index, labels=labels))
    assert_type(labelled_loader, BatchLoader[TrainingBatch])
    for batch in labelled_loader:
        assert_type(module.training_step(batch, 0), torch.Tensor)
