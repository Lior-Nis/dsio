"""StoredItems maps stored samples into exactly the declared training fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dsio.data.adapters import entity_examples
from dsio.data.loading import LoadingError
from dsio.data.store import SignalStore
from dsio.experimental.data import StoredItems


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    with SignalStore.builder(tmp_path / "items", channels=3, dtype="uint8") as builder:
        builder.add("a", np.array([[1, 2, 3], [4, 5, 6]]), group="g1", attrs={"target": 2})
        builder.add("b", np.array([[7, 8, 9], [10, 11, 12]]), group="g2", attrs={"target": 5})
    return SignalStore(tmp_path / "items")


def _item(store: SignalStore, sample_id: str = "a", **fields: object) -> dict[str, torch.Tensor]:
    return StoredItems(**fields)(store, entity_examples(store), [sample_id])[0]  # type: ignore[arg-type]


def test_only_declared_fields_are_produced(store: SignalStore) -> None:
    item = _item(store, x={"from": "data"})
    assert set(item) == {"sample_id", "x"}
    assert item["x"].dtype == torch.float32
    assert item["x"].tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_columns_layout_dtype_and_divide(store: SignalStore) -> None:
    item = _item(
        store,
        x={"from": "data", "columns": [1, 3], "layout": "channel_first", "divide": 255},
        mask={"from": "data", "columns": [0, 1], "dtype": "bool", "shape": [2]},
    )
    expected = torch.tensor([[2.0, 5.0], [3.0, 6.0]]) / 255.0
    assert torch.equal(item["x"], expected)
    assert item["mask"].dtype == torch.bool and item["mask"].tolist() == [True, True]


def test_attribute_target_transforms_apply_in_order(store: SignalStore) -> None:
    scalar = _item(store, x={"from": "data"}, y={"from": "attribute", "attribute": "target"})
    assert scalar["y"].shape == () and scalar["y"].item() == 2.0

    shaped = _item(
        store,
        x={"from": "data"},
        y={"from": "attribute", "attribute": "target", "offset": -1, "log1p": True, "shape": [1]},
    )
    assert shaped["y"].shape == (1,)
    assert torch.equal(shaped["y"], torch.from_numpy(np.log1p(np.array([1.0], np.float32))))

    index = _item(
        store,
        "b",
        x={"from": "data"},
        y={"from": "attribute", "attribute": "target", "dtype": "int64"},
    )
    assert index["y"].dtype == torch.int64 and index["y"].item() == 5


def test_numpy_float32_divide_matches_torch_division_bit_for_bit(store: SignalStore) -> None:
    item = _item(store, x={"from": "data", "divide": 255.0})
    reference = (
        torch.from_numpy(np.array(store.read_sample("a")["data"], copy=True)).float() / 255.0
    )
    assert torch.equal(item["x"], reference)


def test_missing_declared_attribute_and_columns_fail_by_name(store: SignalStore) -> None:
    with pytest.raises(LoadingError, match="field 'y' reads attribute 'label'"):
        _item(store, x={"from": "data"}, y={"from": "attribute", "attribute": "label"})
    with pytest.raises(LoadingError, match="field 'x' reads columns"):
        _item(store, x={"from": "data", "columns": [2, 5]})


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"from": "file"}, "'from' must be"),
        ({"from": "data", "attribute": "target"}, "unsupported keys"),
        ({"from": "attribute"}, "must name its 'attribute'"),
        ({"from": "data", "dtype": "float16"}, "dtype must be"),
        ({"from": "data", "layout": "batch_first"}, "layout must be"),
        ({"from": "data", "columns": [2, 1]}, "columns must be"),
        ({"from": "data", "divide": 0}, "divide must be non-zero"),
        ({"from": "data", "log1p": "yes"}, "log1p must be"),
    ],
)
def test_invalid_field_specifications_fail_at_construction(
    spec: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StoredItems(x=spec)


def test_examples_must_describe_the_store_and_samples_must_exist(
    store: SignalStore, tmp_path: Path
) -> None:
    items = StoredItems(x={"from": "data"})
    examples = entity_examples(store)
    with SignalStore.builder(tmp_path / "other", channels=3, dtype="uint8") as builder:
        builder.add("z", np.zeros((1, 3)), group="g", attrs={})
    with pytest.raises(LoadingError, match="examples describe"):
        items(SignalStore(tmp_path / "other"), examples, ["z"])
    with pytest.raises(LoadingError, match="cannot resolve assigned sample 'missing'"):
        items(store, examples, ["missing"])


def test_a_tampered_payload_fails_the_digest_check(store: SignalStore) -> None:
    dataset = StoredItems(x={"from": "data"})(store, entity_examples(store), ["a"])
    payload = next(Path(store.path).rglob("*.bin"))
    raw = bytearray(payload.read_bytes())
    raw[0] ^= 0xFF
    payload.write_bytes(bytes(raw))
    with pytest.raises(Exception, match="digest"):
        dataset[0]


def test_optional_attribute_fields_are_omitted_for_unlabelled_samples(tmp_path: Path) -> None:
    with SignalStore.builder(tmp_path / "mixed", channels=1) as builder:
        builder.add("train", np.zeros((1, 1)), group="g1", attrs={"target": 4})
        builder.add("test", np.zeros((1, 1)), group="g2", attrs={})
    store = SignalStore(tmp_path / "mixed")
    items = StoredItems(
        x={"from": "data"},
        y={"from": "attribute", "attribute": "target", "dtype": "int64", "optional": True},
    )
    dataset = items(store, entity_examples(store), ["train", "test"])

    assert dataset[0]["y"].item() == 4
    assert "y" not in dataset[1]
    with pytest.raises(ValueError, match="cannot be optional"):
        StoredItems(x={"from": "attribute", "attribute": "target", "optional": True})
