"""Static contracts for dataset items and collated model batches."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NotRequired, Protocol, TypedDict, TypeVar

from torch import Tensor


class WindowItem(TypedDict):
    """One indexed window before collation."""

    x: Tensor
    row: int
    y: NotRequired[Tensor]


class TrainingItem(TypedDict):
    """One indexed window with the target required by model steps."""

    x: Tensor
    row: int
    y: Tensor


class BatchInputs(TypedDict):
    """Fields shared by every collated input batch."""

    x: Tensor
    row: Tensor


class WindowBatch(BatchInputs):
    """A general collated batch, which may be unlabelled."""

    y: NotRequired[Tensor]


class TrainingBatch(BatchInputs):
    """A batch accepted by model steps, where the target is required."""

    y: Tensor


class PredictionBatch(TypedDict):
    """One model prediction batch returned by Lightning."""

    row: Tensor
    prediction: Tensor


LoaderBatch = WindowBatch | TrainingBatch

BatchT_co = TypeVar("BatchT_co", covariant=True)


class BatchLoader(Protocol[BatchT_co]):
    """The batch-level behavior consumers need from a loader."""

    def __iter__(self) -> Iterator[BatchT_co]: ...
