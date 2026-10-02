"""Evaluation and inference arrays assembled through the training dataset and collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import default_collate

from dsio.config.components import resolve_component
from dsio.data.adapters import TableExamples
from dsio.data.examples import Examples
from dsio.data.loading import DatasetFactory, LoadingError
from dsio.data.loading.collation import Collate
from dsio.data.store import SignalStore


def collate_arrays(
    dataset: DatasetFactory | Mapping[str, Any],
    store: SignalStore,
    sample_ids: Sequence[str],
    *,
    examples: Examples | None = None,
    collate_fn: Collate | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Collate the given samples, in order, into NumPy arrays as training would batch them.

    The dataset factory and collation are the ones ``DsioDataModule`` uses (``None`` is
    PyTorch's default collation, as there), so evaluation and inference inputs, targets
    and masks are exactly what a training batch of the same samples holds, and no consumer
    re-implements batching in NumPy.

    Consumes:
        A dataset factory called as ``(store, examples, sample_ids)``, or its component
        configuration (e.g. a ``StoredItems`` config, the same one training records), the
        store, and the samples in the order the arrays must follow.

    Produces:
        ``{"sample_id": str array, <field>: array, ...}``: one entry per collated field,
        each with the samples along the first axis, in the requested order.

    Parameters:
        ``examples``: the store's ``Examples`` (default: its identity without attributes);
        ``collate_fn``: the training collation (default: PyTorch's ``default_collate``).

    Devices:
        CPU; arrays are NumPy copies of the collated CPU tensors.

    Limitations:
        All samples are collated as one batch, so a padding collation pads to the longest
        requested sample, which can differ from the training batches' padding. Collated
        values must be tensors (or the ``sample_id`` strings).

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> import numpy as np
        >>> path = Path(mkdtemp()) / "store"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     _ = builder.add("a", np.array([[1.0]]), group="a", attrs={"target": 0})
        ...     _ = builder.add("b", np.array([[2.0]]), group="b", attrs={"target": 1})
        >>> store = SignalStore(path)
        >>> items = {
        ...     "reference": "dsio.experimental.data.items:StoredItems",
        ...     "parameters": {
        ...         "x": {"from": "data"},
        ...         "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
        ...     },
        ... }
        >>> arrays = collate_arrays(items, store, ["b", "a"])
        >>> arrays["sample_id"].tolist(), arrays["x"].tolist(), arrays["y"].tolist()
        (['b', 'a'], [[[2.0]], [[1.0]]], [1, 0])
    """
    ids = list(sample_ids)
    if not ids:
        raise LoadingError("collate_arrays needs at least one sample")
    if len(set(ids)) != len(ids):
        raise LoadingError("collate_arrays sample ids must be unique")
    factory = resolve_component(dataset) if isinstance(dataset, Mapping) else dataset
    if not callable(factory):
        raise LoadingError("collate_arrays needs a dataset factory or its configuration")
    built = factory(store, _identity_examples(store) if examples is None else examples, ids)
    batch = (collate_fn or default_collate)([built[index] for index in range(len(ids))])
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in batch.items():
        if isinstance(value, Tensor):
            arrays[name] = value.numpy()
        elif name == "sample_id":
            arrays[name] = np.asarray(value, dtype=np.str_)
        else:
            raise LoadingError(f"collated field {name!r} is a {type(value).__name__}, not a tensor")
    if arrays.get("sample_id") is None or arrays["sample_id"].tolist() != ids:
        raise LoadingError("the collated batch does not preserve the requested sample ids")
    return arrays


def _identity_examples(store: SignalStore) -> TableExamples:
    """The store's identity without its attributes, which a dataset factory never reads."""
    entities = list(store.entities)
    return TableExamples(
        name=str(store.path.name),
        sample_ids=[entity.entity_id for entity in entities],
        groups=[entity.group for entity in entities],
        digest=store.identity,
    )


__all__ = ["collate_arrays"]
