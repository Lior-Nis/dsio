"""The Examples protocol: what dsio needs from a dataset, and nothing more.

The point of these tests is that the split layer works on a dataset with no features at
all. If any of them needed a store, a window or a tensor, the abstraction would have leaked.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsio.data import (
    Examples,
    ExamplesError,
    KeyedExamples,
    TableExamples,
    assert_consistent,
    group_attribute,
)
from dsio.data.examples import check
from dsio.splits import SplitFile, folds_from_splits, resolve


@pytest.fixture
def table() -> TableExamples:
    """Twelve rows in four groups, with one numeric and one categorical attribute."""
    return TableExamples(
        name="rows",
        groups=[f"g{i // 3}" for i in range(12)],
        attributes={
            "score": np.arange(12, dtype=float),
            "site": ["A" if i // 3 < 2 else "B" for i in range(12)],
        },
    )


# --- the protocol ---------------------------------------------------------------------


def test_the_adapters_satisfy_the_protocol(table: TableExamples) -> None:
    assert isinstance(table, Examples)
    assert isinstance(KeyedExamples(name="k", keys=["a", "b"]), Examples)


def test_a_non_dataset_is_rejected_with_a_useful_message() -> None:
    """The common mistake is passing the features themselves."""
    with pytest.raises(ExamplesError, match="identity and grouping, not"):
        check(np.zeros((10, 3)))


def test_misaligned_arrays_are_caught_at_the_boundary() -> None:
    """A groups array one short shifts every assignment from that point on, producing a
    perfectly plausible split whose leakage boundary is wrong."""
    broken = TableExamples(
        name="bad", groups=["a", "b"], attributes={"x": np.arange(5, dtype=float)}
    )
    with pytest.raises(ExamplesError, match="has 5 rows for 2 examples"):
        assert_consistent(broken)


def test_a_dataset_without_a_clock_says_so(table: TableExamples) -> None:
    """Returning None is what makes purged splitting unavailable rather than silently
    wrong on data that has an order but no meaningful time."""
    assert table.times() is None


def test_times_are_checked_when_present() -> None:
    backwards = TableExamples(
        name="t",
        groups=["a", "b"],
        times=(np.array([5.0, 1.0]), np.array([1.0, 2.0])),
    )
    with pytest.raises(ExamplesError, match="end before they start"):
        assert_consistent(backwards)


# --- attributes -------------------------------------------------------------------------


def test_an_unknown_attribute_lists_what_exists(table: TableExamples) -> None:
    with pytest.raises(ExamplesError, match="score, site"):
        table.attribute("nope")


def test_group_attribute_averages_a_numeric_key(table: TableExamples) -> None:
    assert group_attribute(table, "score")["g0"] == pytest.approx(1.0)


def test_group_attribute_refuses_a_categorical_that_varies_within_a_group() -> None:
    """Averaging a site code across a group that moved between sites produces a number
    that means nothing and balances nothing."""
    mixed = TableExamples(
        name="m", groups=["a", "a"], attributes={"site": ["X", "Y"]}
    )
    with pytest.raises(ExamplesError, match="must be constant per group"):
        group_attribute(mixed, "site")


# --- subsetting ---------------------------------------------------------------------------


def test_subset_keeps_every_parallel_array_aligned(table: TableExamples) -> None:
    mask = np.array([i < 6 for i in range(12)])
    part = table.subset(mask)
    assert len(part) == 6
    assert set(part.groups.tolist()) == {"g0", "g1"}
    assert part.attribute("score").tolist() == list(range(6))


def test_subset_preserves_identity(table: TableExamples) -> None:
    """A subset is still the same dataset, so a split file still binds to it."""
    part = table.subset(np.ones(12, dtype=bool))
    assert part.name == table.name and part.digest == table.digest


def test_the_digest_reflects_the_grouping_not_the_features() -> None:
    """Two tables that divide the same way are the same division, so one split serves both."""
    a = TableExamples(name="x", groups=["a", "a", "b"], attributes={"k": [1.0, 2.0, 3.0]})
    b = TableExamples(name="x", groups=["a", "a", "b"], attributes={"k": [1.0, 2.0, 3.0]})
    c = TableExamples(name="x", groups=["a", "b", "b"], attributes={"k": [1.0, 2.0, 3.0]})
    assert a.digest == b.digest
    assert a.digest != c.digest


# --- keyed examples --------------------------------------------------------------------------


def test_keys_default_to_being_their_own_group() -> None:
    """An explicit statement that the records are independent, rather than a silent
    assumption of it."""
    keyed = KeyedExamples(name="docs", keys=["d1", "d2", "d3"])
    assert keyed.groups.tolist() == ["d1", "d2", "d3"]


def test_keys_survive_subsetting() -> None:
    """Predictions join back to the record they came from by name, not by position."""
    keyed = KeyedExamples(name="docs", keys=["d1", "d2", "d3"], groups=["c", "c", "d"])
    part = keyed.subset(np.array([True, False, True]))
    assert part.keys.tolist() == ["d1", "d3"]
    assert part.groups.tolist() == ["c", "d"]


# --- the whole split layer, on data with no features at all -----------------------------------


def _table_kfold(table: TableExamples) -> list[SplitFile]:
    """Two hand-picked folds over the table's four groups, each covering all of them."""
    return [
        SplitFile(
            store=table.name,
            store_manifest_sha256=table.digest,
            name="k2",
            fold=0,
            counts={"train": 6, "test": 6},
            parts={"train": ["g2", "g3"], "test": ["g0", "g1"]},
        ),
        SplitFile(
            store=table.name,
            store_manifest_sha256=table.digest,
            name="k2",
            fold=1,
            counts={"train": 6, "test": 6},
            parts={"train": ["g0", "g1"], "test": ["g2", "g3"]},
        ),
    ]


def test_splitting_works_on_a_dataset_that_is_only_grouping(table: TableExamples) -> None:
    """The headline: no store, no windows, no tensors — and the split layer does not care."""
    splits = _table_kfold(table)
    assert len(splits) == 2
    parts = resolve(table, splits[0])
    assert set(parts) >= {"train", "test"}
    assert sum(len(part) for part in parts.values()) == len(table)


def test_folds_build_from_a_plain_table(table: TableExamples) -> None:
    splits = _table_kfold(table)
    folds = folds_from_splits(table, splits)
    assert len(folds) == 2
    assert sum(fold.test.size for fold in folds) == len(table)
    groups = table.groups
    for fold in folds:
        assert not (set(groups[fold.train]) & set(groups[fold.test]))


def test_a_table_cannot_prove_row_overlap_and_says_why(table: TableExamples) -> None:
    """The signal-specific check is not part of the protocol, and asking for it on a
    dataset with nothing underneath to overlap must explain rather than crash."""
    from dsio.splits import assert_no_row_overlap

    splits = _table_kfold(table)
    parts = resolve(table, splits[0])
    with pytest.raises(Exception, match="no\n *covered_rows|covered_rows"):
        assert_no_row_overlap(parts)


def test_a_keyed_dataset_splits_by_a_coarser_group() -> None:
    """Documents grouped by conversation: the leakage boundary is not the record."""
    keyed = KeyedExamples(
        name="turns",
        keys=[f"t{i}" for i in range(12)],
        groups=[f"conv{i // 4}" for i in range(12)],
    )
    # One hand-picked group held out per fold — the trivial leave-one-group-out shape.
    splits = [
        SplitFile(
            store=keyed.name,
            store_manifest_sha256=keyed.digest,
            name="k3",
            fold=i,
            counts={"train": 8, "test": 4},
            parts={
                "train": [g for g in ("conv0", "conv1", "conv2") if g != held],
                "test": [held],
            },
        )
        for i, held in enumerate(("conv0", "conv1", "conv2"))
    ]
    folds = folds_from_splits(keyed, splits)
    for fold in folds:
        assert not (set(keyed.groups[fold.train]) & set(keyed.groups[fold.test]))
