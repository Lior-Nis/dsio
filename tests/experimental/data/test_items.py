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


def test_declared_value_constraints_run_before_lossy_dtype_casts(tmp_path: Path) -> None:
    path = tmp_path / "constrained"
    with SignalStore.builder(path, channels=3) as builder:
        builder.add(
            "bad",
            np.array([[1.0, np.nan, 0.5]], dtype=np.float32),
            group="g",
        )
    constrained = SignalStore(path)
    examples = entity_examples(constrained)
    items = StoredItems(
        x={"from": "data", "columns": [0, 1]},
        y={"from": "data", "columns": [1, 2], "finite": True},
        mask={
            "from": "data",
            "columns": [2, 3],
            "dtype": "bool",
            "allowed_values": [0, 1],
        },
    )(constrained, examples, ["bad"])

    with pytest.raises(LoadingError, match="field 'y'.*finite"):
        _ = items[0]

    mask_only = StoredItems(
        x={"from": "data", "columns": [0, 1]},
        mask={
            "from": "data",
            "columns": [2, 3],
            "dtype": "bool",
            "allowed_values": [0, 1],
        },
    )(constrained, examples, ["bad"])
    with pytest.raises(LoadingError, match="field 'mask'.*allowed values"):
        _ = mask_only[0]


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
        ({"from": "data", "finite": "yes"}, "finite must be"),
        ({"from": "data", "allowed_values": []}, "allowed_values must be"),
        ({"from": "data", "allowed_values": [float("nan")]}, "allowed_values must be"),
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


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"from": "data", "dtype": "bool", "offset": 1}, "a bool field accepts no"),
        ({"from": "data", "dtype": "int64", "log1p": True}, "accepts only an integer offset"),
        ({"from": "data", "dtype": "int64", "divide": 2}, "accepts only an integer offset"),
        ({"from": "data", "dtype": "int64", "offset": 0.5}, "accepts only an integer offset"),
        ({"from": "data", "shape": [-1, -1]}, "at most one -1"),
        ({"from": "data", "shape": [True]}, "at most one -1"),
        ({"from": ["data"]}, "'from' must be"),
        ({"from": "data", "dtype": ["float32"]}, "dtype must be"),
    ],
)
def test_transforms_must_preserve_the_declared_dtype(spec: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StoredItems(x=spec)


def test_integer_offsets_keep_int64_and_float_transforms_keep_float32(store: SignalStore) -> None:
    item = _item(
        store,
        x={"from": "data", "divide": 3},
        y={"from": "attribute", "attribute": "target", "dtype": "int64", "offset": -1},
    )
    assert item["x"].dtype == torch.float32
    assert item["y"].dtype == torch.int64 and item["y"].item() == 1


def test_null_attributes_bad_reshapes_and_excess_columns_name_the_field(
    tmp_path: Path, store: SignalStore
) -> None:
    with SignalStore.builder(tmp_path / "null", channels=1) as builder:
        builder.add("n", np.zeros((1, 1)), group="g", attrs={"target": None})
    nulls = SignalStore(tmp_path / "null")
    with pytest.raises(LoadingError, match="field 'y' reads attribute 'target', which is null"):
        StoredItems(x={"from": "data"}, y={"from": "attribute", "attribute": "target"})(
            nulls, entity_examples(nulls), ["n"]
        )[0]
    with pytest.raises(LoadingError, match="field 'x' cannot reshape sample 'a'"):
        _item(store, x={"from": "data", "shape": [4]})
    with pytest.raises(LoadingError, match="field 'x' reads columns \\[0, 9\\] but store"):
        StoredItems(x={"from": "data", "columns": [0, 9]})(store, entity_examples(store), ["a"])
