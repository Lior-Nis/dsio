"""The warehouse padding collator owns ragged batching without task-specific modes."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from dsio.config.components import resolve_component
from dsio.data.loading import LoadingError
from dsio.data.loading.collation import collate_items
from dsio.experimental.data import PadCollator


def test_declared_fields_are_padded_stacked_and_masked_in_identity_order() -> None:
    collator = PadCollator(
        padded_fields={"x": -1.0, "sequence_y": 9},
        fixed_fields=["y", "weight"],
        emit_mask=True,
    )
    items = [
        {
            "sample_id": "short",
            "x": np.ones((2, 3), dtype=np.float32),
            "sequence_y": torch.tensor([1, 2], dtype=torch.int64),
            "y": torch.tensor(4),
            "weight": np.float32(0.25),
        },
        {
            "sample_id": "long",
            "x": np.full((4, 3), 2, dtype=np.float32),
            "sequence_y": torch.tensor([3, 4, 5, 6], dtype=torch.int64),
            "y": torch.tensor(5),
            "weight": np.float32(0.75),
        },
    ]

    batch = collate_items(items, collate_fn=collator)

    assert batch["sample_id"] == ["short", "long"]
    assert batch["x"].dtype == torch.float32
    assert batch["sequence_y"].dtype == torch.int64
    assert batch["weight"].dtype == torch.float32
    assert tuple(batch["x"].shape) == (2, 4, 3)
    assert batch["x"][0, 2:].eq(-1).all()
    assert batch["sequence_y"][0].tolist() == [1, 2, 9, 9]
    assert batch["y"].tolist() == [4, 5]
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_component_config_resolves_to_a_pickle_safe_collator() -> None:
    config = {
        "reference": "dsio.experimental.data.padding:PadCollator",
        "parameters": {
            "padded_fields": {"x": 0, "y": 0},
            "fixed_fields": [],
            "emit_mask": True,
        },
    }

    collator = resolve_component(config, expected=PadCollator)

    restored = pickle.loads(pickle.dumps(collator))
    batch = restored(
        [
            {"sample_id": "a", "x": torch.ones(1, 2), "y": torch.ones(1)},
            {"sample_id": "b", "x": torch.ones(2, 2), "y": torch.ones(2)},
        ]
    )
    assert batch["mask"].tolist() == [[True, False], [True, True]]


def test_declared_fixed_fields_may_be_absent_from_an_inputs_only_batch() -> None:
    batch = PadCollator(padded_fields={"x": 0}, fixed_fields=["y"])(
        [
            {"sample_id": "a", "x": torch.ones(1)},
            {"sample_id": "b", "x": torch.ones(2)},
        ]
    )

    assert set(batch) == {"sample_id", "x"}


def test_declared_fixed_fields_cannot_be_present_in_only_part_of_a_batch() -> None:
    with pytest.raises(LoadingError, match="fixed field 'y'.*every item or none"):
        PadCollator(padded_fields={"x": 0}, fixed_fields=["y"])(
            [
                {"sample_id": "a", "x": torch.ones(1), "y": 1},
                {"sample_id": "b", "x": torch.ones(2)},
            ]
        )


def test_supplied_domain_mask_keeps_real_false_values_and_pads_false() -> None:
    batch = PadCollator(padded_fields={"x": 0.0, "mask": False})(
        [
            {
                "sample_id": "short",
                "x": torch.ones(2, 1),
                "mask": torch.tensor([False, True]),
            },
            {
                "sample_id": "long",
                "x": torch.ones(3, 1),
                "mask": torch.tensor([True, True, False]),
            },
        ]
    )

    assert batch["mask"].tolist() == [[False, True, False], [True, True, False]]


@pytest.mark.parametrize(
    ("right", "message"),
    [
        (torch.ones(3, 3), "trailing shapes"),
        (torch.ones(3, 2, dtype=torch.int64), "dtypes"),
    ],
)
def test_padded_field_shapes_and_dtypes_must_match(right: torch.Tensor, message: str) -> None:
    with pytest.raises(LoadingError, match=message):
        PadCollator(padded_fields={"x": 0})(
            [
                {"sample_id": "a", "x": torch.ones(2, 2)},
                {"sample_id": "b", "x": right},
            ]
        )


@pytest.mark.parametrize("padding", [0.5, 2**80])
def test_integer_padding_must_be_exactly_representable(padding: float | int) -> None:
    with pytest.raises(LoadingError, match="padding value.*represented.*int64"):
        PadCollator(padded_fields={"x": padding})(
            [
                {"sample_id": "a", "x": torch.ones(1, dtype=torch.int64)},
                {"sample_id": "b", "x": torch.ones(2, dtype=torch.int64)},
            ]
        )


def test_boolean_padding_accepts_only_boolean_values() -> None:
    with pytest.raises(LoadingError, match="padding value.*represented.*bool"):
        PadCollator(padded_fields={"mask": 2})(
            [
                {"sample_id": "a", "mask": torch.ones(1, dtype=torch.bool)},
                {"sample_id": "b", "mask": torch.ones(2, dtype=torch.bool)},
            ]
        )


def test_fixed_fields_must_remain_on_cpu() -> None:
    with pytest.raises(LoadingError, match="fixed field 'y'.*CPU"):
        PadCollator(padded_fields={"x": 0}, fixed_fields=["y"])(
            [
                {
                    "sample_id": "a",
                    "x": torch.ones(1),
                    "y": torch.ones(1, device="meta"),
                }
            ]
        )


def test_alignment_error_names_the_sample_fields_and_lengths() -> None:
    collator = PadCollator(padded_fields={"x": 0, "y": 0, "mask": False})

    with pytest.raises(
        LoadingError,
        match=r"sample 'broken'.*x=3.*y=2.*mask=4",
    ):
        collator(
            [
                {
                    "sample_id": "broken",
                    "x": torch.ones(3, 2),
                    "y": torch.ones(2),
                    "mask": torch.ones(4, dtype=torch.bool),
                }
            ]
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PadCollator(padded_fields={}), "at least one padded field"),
        (
            lambda: PadCollator(padded_fields={"sample_id": 0}),
            "sample_id",
        ),
        (
            lambda: PadCollator(padded_fields={"x": 0}, fixed_fields=["x"]),
            "both padded and fixed",
        ),
        (
            lambda: PadCollator(padded_fields={"x": 0, "mask": False}, emit_mask=True),
            "mask",
        ),
    ],
)
def test_invalid_configuration_fails_at_construction(factory, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(LoadingError, match=message):
        factory()


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "empty batch"),
        ([{"sample_id": "a", "x": torch.ones(2), "extra": 1}], "undeclared.*extra"),
        ([{"sample_id": "a"}], "missing.*x"),
        ([{"sample_id": "a", "x": 1}], "padded field 'x'.*axis zero"),
        ([{"sample_id": "a", "x": ["bad"]}], "padded field 'x'.*tensor"),
    ],
)
def test_invalid_items_fail_by_name(items: list[dict[str, object]], message: str) -> None:
    with pytest.raises(LoadingError, match=message):
        PadCollator(padded_fields={"x": 0})(items)
