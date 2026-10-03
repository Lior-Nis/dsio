"""Evaluation and inference arrays assembled through the training dataset and collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from dsio.config.components import resolve_component
from dsio.data.adapters import TableExamples
from dsio.data.examples import Examples
from dsio.data.loading import DatasetFactory, IdentityDataset, LoadingError
from dsio.data.loading.collation import Collate, collate_items
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

    The factory and collation run behind the same guard ``DsioDataModule`` applies
    (``None`` is PyTorch's default collation), so with the training dataset configuration
    the arrays are exactly what a training batch of the same samples holds, and no
    consumer re-implements batching in NumPy. Evaluation and inference may declare their
    own field specs (e.g. raw ``uint8`` inputs, or no target) over the same factory.

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
        CPU; arrays are NumPy views of the collated CPU tensors.

    Limitations:
        All samples are collated as one batch, so a padding collation pads to the longest
        requested sample, which can differ from the training batches' padding. Collated
        values must be tensors or arrays. Sample ids must be unique. A training dataset
        that declares ``y`` cannot serve inference on samples without that target; declare
        an inputs-only config. The default examples carry no attributes, so a factory that
        reads them needs ``examples=``. The whole roster is held in memory as one batch,
        and arrays from tensors share their memory.

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
    requested_ids = tuple(sample_ids)
    if not requested_ids:
        raise LoadingError("collate_arrays needs at least one sample")
    if len(set(requested_ids)) != len(requested_ids):
        raise LoadingError("collate_arrays sample ids must be unique")
    factory = resolve_component(dataset) if isinstance(dataset, Mapping) else dataset
    if not callable(factory):
        raise LoadingError("collate_arrays needs a dataset factory or its configuration")
    try:
        built = factory(
            store,
            _identity_examples(store) if examples is None else examples,
            list(requested_ids),
        )
    except LoadingError:
        raise
    except Exception as error:
        raise LoadingError(f"dataset factory failed: {error}") from error
    if not isinstance(built, Dataset):
        raise LoadingError(
            f"dataset factory must return a torch Dataset, got {type(built).__name__}"
        )
    if isinstance(built, IterableDataset):
        raise LoadingError("dataset factory must return a map-style Dataset, not IterableDataset")
    guarded = IdentityDataset(built, requested_ids)
    batch = collate_items(
        [guarded[index] for index in range(len(requested_ids))], collate_fn=collate_fn
    )
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "sample_id": np.asarray(batch["sample_id"], dtype=np.str_)
    }
    for name, value in batch.items():
        if name == "sample_id":
            continue
        if isinstance(value, Tensor):
            arrays[name] = value.numpy()
        elif isinstance(value, np.ndarray):
            arrays[name] = value
        else:
            raise LoadingError(f"collated field {name!r} is a {type(value).__name__}, not an array")
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
