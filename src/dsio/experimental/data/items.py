"""Map stored samples into identity-bearing training items, field by field."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dsio.data.examples import Examples
from dsio.data.loading.datasets import LoadingError
from dsio.data.store import SignalStore, StoreError

FIELDS = ("x", "y", "mask", "sample_weight")
_DTYPES = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "bool": np.bool_}
_LAYOUTS = ("time_major", "channel_first")
_KEYS = {
    "data": {"from", "columns", "dtype", "layout", "offset", "log1p", "divide", "shape"},
    "attribute": {"from", "attribute", "optional", "dtype", "offset", "log1p", "divide", "shape"},
}


class StoredItems:
    """Map stored samples into identity-bearing training items, field by field.

    A dataset factory: an instance is passed as ``DsioDataModule(dataset_factory=...)`` and
    called with ``(store, examples, sample_ids)``. Each declared field reads either the
    sample's stored data array (optionally a column range) or one entity attribute, casts it,
    applies the declared numeric transforms in a fixed order (offset, log1p, divide), and
    reshapes it. Undeclared fields are never invented; an unlabelled dataset simply declares
    no ``y``, and an ``optional`` attribute field is omitted for samples without it (e.g.
    unlabelled test rows sharing a store with labelled training rows).

    Consumes:
        A ``SignalStore`` whose samples are ``[rows, channels]`` arrays with entity
        attributes, the ``Examples`` that describe that exact store (name and content
        digest), and the assigned ``sample_ids``.

    Produces:
        Items ``{"sample_id", "x", ...}`` with exactly the declared fields among ``x``,
        ``y``, ``mask`` and ``sample_weight``, as CPU tensors.

    Parameters:
        One mapping per field (``x`` required):

        - ``from``: ``"data"`` or ``"attribute"``; ``attribute``: the entity attribute name;
          ``optional`` (attribute fields only, default ``false``) omits the field for
          samples that lack the attribute.
        - ``columns``: ``[start, stop]`` of the data array; ``layout``: ``time_major``
          (default, as stored) or ``channel_first`` (transposed to ``[channels, rows]``).
        - ``dtype``: ``float32`` (default), ``float64``, ``int64`` or ``bool``.
        - ``offset`` (added), ``log1p`` (``true``) and ``divide`` (divisor), applied in
          that order after the cast; ``shape``: final shape, e.g. ``[1]`` or ``[]``.

    Devices:
        CPU; items are moved to the accelerator by Lightning after collation.

    Limitations:
        Whole-sample reads only (windows are a separate block); numeric transforms run in
        NumPy on the cast array, so they match NumPy-based preprocessing bit for bit.

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from dsio.data.adapters import entity_examples
        >>> from dsio.data.store import SignalStore
        >>> from tempfile import mkdtemp
        >>> path = Path(mkdtemp()) / "store"
        >>> with SignalStore.builder(path, channels=2) as builder:
        ...     _ = builder.add("a", np.array([[1.0, 2.0]]), group="g1", attrs={"target": 3.0})
        ...     _ = builder.add("b", np.array([[4.0, 5.0]]), group="g2", attrs={"target": 6.0})
        >>> store = SignalStore(path)
        >>> items = StoredItems(
        ...     x={"from": "data", "dtype": "float32"},
        ...     y={"from": "attribute", "attribute": "target", "log1p": True, "shape": [1]},
        ... )
        >>> dataset = items(store, entity_examples(store), ["b"])
        >>> item = dataset[0]
        >>> item["sample_id"], item["x"].tolist(), round(float(item["y"][0]), 4)
        ('b', [[4.0, 5.0]], 1.9459)
    """

    def __init__(
        self,
        *,
        x: Mapping[str, Any],
        y: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
        sample_weight: Mapping[str, Any] | None = None,
    ) -> None:
        declared = {"x": x, "y": y, "mask": mask, "sample_weight": sample_weight}
        self.fields = {
            name: _field(name, spec) for name, spec in declared.items() if spec is not None
        }

    def __call__(
        self, store: SignalStore, examples: Examples, sample_ids: Sequence[str]
    ) -> Dataset[dict[str, Any]]:
        if examples.name != store.path.name or examples.digest != store.identity:
            raise LoadingError(
                f"stored-item factory received store {store.path.name!r} with content identity "
                f"{store.identity!r}, but examples describe {examples.name!r} with "
                f"{examples.digest!r}"
            )
        return _StoredItemsDataset(store, sample_ids, self.fields)


class _StoredItemsDataset(Dataset[dict[str, Any]]):
    """The map-style dataset a :class:`StoredItems` factory builds for one role."""

    def __init__(
        self,
        store: SignalStore,
        sample_ids: Sequence[str],
        fields: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.store = store
        self.sample_ids = tuple(sample_ids)
        self.fields = dict(fields)
        for sample_id in self.sample_ids:
            try:
                store.entity(sample_id)
            except StoreError as error:
                raise LoadingError(
                    f"stored-item dataset cannot resolve assigned sample {sample_id!r}: {error}"
                ) from error

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, position: int) -> dict[str, Any]:
        sample = self.store.read_sample(self.sample_ids[position])
        item: dict[str, Any] = {"sample_id": sample["sample_id"]}
        for name, spec in self.fields.items():
            if spec.get("optional") and spec["attribute"] not in sample["attrs"]:
                continue
            item[name] = torch.from_numpy(_value(name, spec, sample))
        return item


def _field(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"field {name!r} must be a mapping, got {type(spec).__name__}")
    source = spec.get("from")
    if source not in _KEYS:
        raise ValueError(f"field {name!r}: 'from' must be 'data' or 'attribute', got {source!r}")
    unknown = sorted(set(spec) - _KEYS[source])
    if unknown:
        raise ValueError(f"field {name!r} from {source}: unsupported keys {unknown}")
    if source == "attribute" and not (isinstance(spec.get("attribute"), str) and spec["attribute"]):
        raise ValueError(f"field {name!r}: an attribute field must name its 'attribute'")
    if spec.get("dtype", "float32") not in _DTYPES:
        raise ValueError(f"field {name!r}: dtype must be one of {sorted(_DTYPES)}")
    if spec.get("layout", "time_major") not in _LAYOUTS:
        raise ValueError(f"field {name!r}: layout must be one of {list(_LAYOUTS)}")
    columns = spec.get("columns")
    if columns is not None and not (
        isinstance(columns, Sequence)
        and len(columns) == 2
        and all(isinstance(bound, int) and not isinstance(bound, bool) for bound in columns)
        and 0 <= columns[0] < columns[1]
    ):
        raise ValueError(f"field {name!r}: columns must be [start, stop] with 0 <= start < stop")
    for key in ("offset", "divide"):
        value = spec.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"field {name!r}: {key} must be a finite number")
    if spec.get("divide") == 0:
        raise ValueError(f"field {name!r}: divide must be non-zero")
    for flag in ("log1p", "optional"):
        if not isinstance(spec.get(flag, False), bool):
            raise ValueError(f"field {name!r}: {flag} must be true or false")
    if name == "x" and spec.get("optional"):
        raise ValueError("field 'x' is the model input and cannot be optional")
    shape = spec.get("shape")
    if shape is not None and not (
        isinstance(shape, Sequence) and all(isinstance(size, int) and size >= -1 for size in shape)
    ):
        raise ValueError(f"field {name!r}: shape must be a list of integers")
    return dict(spec)


def _value(name: str, spec: Mapping[str, Any], sample: Mapping[str, Any]) -> np.ndarray[Any, Any]:
    dtype = _DTYPES[spec.get("dtype", "float32")]
    array: np.ndarray[Any, Any]
    if spec["from"] == "data":
        array = np.array(sample["data"], dtype=dtype, copy=True)
        columns = spec.get("columns")
        if columns is not None:
            if array.ndim != 2 or columns[1] > array.shape[1]:
                raise LoadingError(
                    f"field {name!r} reads columns {columns} but sample "
                    f"{sample['sample_id']!r} has shape {tuple(array.shape)}"
                )
            array = np.ascontiguousarray(array[:, columns[0] : columns[1]])
        if spec.get("layout") == "channel_first":
            array = np.ascontiguousarray(array.T)
    else:
        attribute = spec["attribute"]
        attrs = sample["attrs"]
        if attribute not in attrs:
            raise LoadingError(
                f"field {name!r} reads attribute {attribute!r}, which sample "
                f"{sample['sample_id']!r} does not have"
            )
        array = np.array(attrs[attribute], dtype=dtype, copy=True)
    if spec.get("offset") is not None:
        array = array + array.dtype.type(spec["offset"])
    if spec.get("log1p"):
        array = np.log1p(array)
    if spec.get("divide") is not None:
        array = array / spec["divide"]
        if array.dtype != dtype and dtype in (np.float32, np.float64):
            array = array.astype(dtype)
    shape = spec.get("shape")
    if shape is not None:
        array = array.reshape(shape)
    # np.ascontiguousarray would promote a 0-d scalar target to shape (1,).
    return np.asarray(array, order="C")


__all__ = ["StoredItems"]
