"""Evaluation arrays are exactly what the training collation produces for the same samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from dsio.data.adapters import entity_examples
from dsio.data.loading import LoadingError
from dsio.data.store import SignalStore
from dsio.experimental.data import StoredItems, collate_arrays

ITEMS = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "layout": "channel_first"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
    },
}


def _store(path: Path) -> SignalStore:
    with SignalStore.builder(path, channels=2) as builder:
        for index in range(6):
            values = np.arange(6, dtype=np.float32).reshape(3, 2) + 10 * index
            _ = builder.add(f"s{index}", values, group=f"g{index}", attrs={"target": index % 3})
    return SignalStore(path)


def test_arrays_equal_the_training_collation_in_the_requested_order(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    ids = ["s4", "s0", "s5", "s2"]
    arrays = collate_arrays(ITEMS, store, ids)

    factory = StoredItems(**ITEMS["parameters"])  # type: ignore[arg-type]
    loader = torch.utils.data.DataLoader(
        factory(store, entity_examples(store), ids), batch_size=3, shuffle=False
    )
    batches = list(loader)
    assert arrays["sample_id"].tolist() == ids
    for field in ("x", "y"):
        expected = torch.cat([batch[field] for batch in batches]).numpy()
        assert arrays[field].dtype == expected.dtype
        np.testing.assert_array_equal(arrays[field], expected)
    assert arrays["x"].shape == (4, 2, 3)  # channel-first, as training sees it


def test_a_factory_custom_collation_and_raw_pixels_are_supported(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        batch = torch.utils.data.default_collate(items)
        return {**batch, "x": batch["x"] * 2}

    pixels = StoredItems(x={"from": "data", "dtype": "uint8"})
    arrays = collate_arrays(pixels, store, ["s1"], collate_fn=collate)
    assert arrays["x"].dtype == np.uint8
    assert arrays["x"].tolist() == [[[20, 22], [24, 26], [28, 30]]]
    with pytest.raises(ValueError, match="uint8 field accepts no"):
        StoredItems(x={"from": "data", "dtype": "uint8", "divide": 255.0})


def test_inputs_are_checked(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    with pytest.raises(LoadingError, match="at least one sample"):
        collate_arrays(ITEMS, store, [])
    with pytest.raises(LoadingError, match="unique"):
        collate_arrays(ITEMS, store, ["s1", "s1"])
    with pytest.raises(LoadingError, match="cannot resolve assigned sample"):
        collate_arrays(ITEMS, store, ["missing"])

    def strings(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"sample_id": [item["sample_id"] for item in items], "x": ["not a tensor"]}

    with pytest.raises(LoadingError, match="mapping|cardinality|rows|'x'"):
        collate_arrays(ITEMS, store, ["s1"], collate_fn=strings)

    def reordered(items: list[dict[str, Any]]) -> dict[str, Any]:
        return torch.utils.data.default_collate(list(reversed(items)))

    with pytest.raises(LoadingError, match="changed sample_id order"):
        collate_arrays(ITEMS, store, ["s1", "s2"], collate_fn=reordered)


def test_the_training_guard_applies_to_collation(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")

    def arrays(items: list[dict[str, Any]]) -> dict[str, Any]:
        batch = torch.utils.data.default_collate(items)
        return {**batch, "x": batch["x"].numpy()}

    assert isinstance(collate_arrays(ITEMS, store, ["s1"], collate_fn=arrays)["x"], np.ndarray)

    def wrong_rows(items: list[dict[str, Any]]) -> dict[str, Any]:
        batch = torch.utils.data.default_collate(items)
        return {**batch, "y": batch["y"][:1]}

    with pytest.raises(LoadingError):
        collate_arrays(ITEMS, store, ["s1", "s2"], collate_fn=wrong_rows)
    with pytest.raises(LoadingError, match="mapping"):
        collate_arrays(ITEMS, store, ["s1"], collate_fn=lambda items: items)  # type: ignore[arg-type,return-value]


def test_the_training_identity_guard_applies_before_collation(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")

    class Items(Dataset[dict[str, Any]]):
        def __init__(self, items: list[Any]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, position: int) -> Any:
            return self.items[position]

    def reversed_factory(store, examples, sample_ids):  # type: ignore[no-untyped-def]
        del store, examples
        return Items([{"sample_id": value, "x": torch.ones(1)} for value in reversed(sample_ids)])

    with pytest.raises(LoadingError, match="changed sample_id"):
        collate_arrays(reversed_factory, store, ["s1", "s2"])

    def mutating_factory(store, examples, sample_ids):  # type: ignore[no-untyped-def]
        del store, examples
        requested = list(sample_ids)
        sample_ids.reverse()
        return Items([{"sample_id": value, "x": torch.ones(1)} for value in reversed(requested)])

    with pytest.raises(LoadingError, match="changed sample_id"):
        collate_arrays(mutating_factory, store, ["s1", "s2"])

    def short_factory(store, examples, sample_ids):  # type: ignore[no-untyped-def]
        del store, examples, sample_ids
        return Items([{"sample_id": "s1", "x": torch.ones(1)}])

    with pytest.raises(LoadingError, match="returned 1 items"):
        collate_arrays(short_factory, store, ["s1", "s2"])

    def non_mapping_factory(store, examples, sample_ids):  # type: ignore[no-untyped-def]
        del store, examples, sample_ids
        return Items(["not a mapping"])

    with pytest.raises(LoadingError, match="must be a mapping"):
        collate_arrays(non_mapping_factory, store, ["s1"])

    class Stream(IterableDataset[dict[str, Any]]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            yield {"sample_id": "s1", "x": torch.ones(1)}

    def iterable_factory(store, examples, sample_ids):  # type: ignore[no-untyped-def]
        del store, examples, sample_ids
        return Stream()

    with pytest.raises(LoadingError, match="map-style Dataset"):
        collate_arrays(iterable_factory, store, ["s1"])
