"""Split invariants. The leakage tests here are the point of the whole module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data import SignalExamples, SignalStore, WindowSpec, build_index, entity_examples
from dsio.splits import SplitError, SplitFile, assert_no_row_overlap, resolve


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    """Nine groups, two recordings each, with an uneven event count."""
    path = tmp_path / "cohort"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=3) as builder:
        for group in range(9):
            for session in range(2):
                builder.add(
                    f"p{group}_s{session}",
                    rng.standard_normal((1500, 3)).astype("float32"),
                    group=f"p{group}",
                    attrs={"events": group * 10},
                )
    return SignalStore(path)


@pytest.fixture
def index(store: SignalStore):
    return build_index(store, WindowSpec(length=500, stride=200))


def _fold0(store: SignalStore) -> SplitFile:
    """One hand-picked fold covering all nine groups, mutually disjoint across parts.

    A literal fixture, not a generated one — a project's own split generator would write
    something like this file and commit it; the tests below only need one to read.
    """
    return SplitFile(
        store=store.path.name,
        store_manifest_sha256=entity_examples(store).digest,
        name="k3",
        fold=0,
        counts={"train": 6, "val": 1, "test": 2},
        parts={
            "train": ["p2", "p3", "p4", "p5", "p6", "p7"],
            "val": ["p8"],
            "test": ["p0", "p1"],
        },
    )


# --- the check that matters most ----------------------------------------------------


def test_overlapping_parts_are_rejected() -> None:
    """Validating duplicates *within* a part is the obvious check and the insufficient one.

    A group in both train and test passes it and silently invalidates every number the
    split produces.
    """
    with pytest.raises(ValueError, match="mutually disjoint"):
        SplitFile(
            store="s", name="bad", parts={"train": ["p1", "p2"], "test": ["p2", "p3"]}
        )


def test_duplicates_within_a_part_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        SplitFile(store="s", name="bad", parts={"train": ["p1", "p1"]})


def test_split_with_neither_groups_nor_time_is_rejected() -> None:
    """A split must divide something; empty is not a valid partition."""
    with pytest.raises(ValueError, match="group parts, temporal bounds, or both"):
        SplitFile(store="s", name="bad", parts={})


def test_no_raw_row_appears_in_two_parts(store: SignalStore, index) -> None:
    """The structural guarantee, verified directly rather than assumed."""
    assert_no_row_overlap(resolve(SignalExamples(store, index), _fold0(store)))


def test_row_overlap_is_detectable_when_it_exists(store: SignalStore, index) -> None:
    """The detector must actually detect; a check that never fires proves nothing."""
    groups = index.groups
    left = index.subset(np.isin(groups, ["p0", "p1"]))
    overlapping = index.subset(np.isin(groups, ["p1", "p2"]))
    with pytest.raises(SplitError, match="share .* raw row"):
        assert_no_row_overlap({"a": left, "b": overlapping})


# --- file round trip ----------------------------------------------------------------


def test_split_file_round_trips(store: SignalStore, tmp_path: Path) -> None:
    split = SplitFile(
        store=store.path.name,
        store_manifest_sha256=entity_examples(store).digest,
        name="k3",
        fold=0,
        counts={"train": 7, "test": 2},
        parts={"train": sorted(f"p{i}" for i in range(2, 9)), "test": ["p0", "p1"]},
    )
    path = tmp_path / "splits" / "k3" / "fold0.yaml"
    split.save(path)
    restored = SplitFile.load(path)
    assert restored.parts == split.parts


def test_foreign_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("schema_version: other/9\nstore: s\nname: n\nparts: {train: [a]}\n")
    with pytest.raises(SplitError, match="expected"):
        SplitFile.load(path)


def test_to_yaml_header_states_counts_and_group_key() -> None:
    """The header is read in a diff without parsing the body, so these two lines — what
    the leakage boundary is, and how big each part is — must actually be there."""
    split = SplitFile(
        store="s",
        name="k3",
        fold=0,
        group_key="subject",
        counts={"train": 2, "test": 1},
        parts={"train": ["a", "b"], "test": ["c"]},
    )
    header = split.to_yaml()
    assert "# group key: subject  <- the leakage boundary" in header
    assert "# counts: test=1, train=2" in header


# --- resolution ---------------------------------------------------------------------


def test_resolve_produces_disjoint_group_sets(store: SignalStore, index) -> None:
    parts = resolve(SignalExamples(store, index), _fold0(store))
    seen: set[str] = set()
    for sub in parts.values():
        groups = set(sub.groups.tolist())
        assert not (groups & seen)
        seen |= groups


def test_resolve_accounts_for_every_window(store: SignalStore, index) -> None:
    parts = resolve(SignalExamples(store, index), _fold0(store))
    assert sum(len(sub) for sub in parts.values()) == len(index)


def test_resolve_rejects_a_split_from_another_store(store: SignalStore, index) -> None:
    foreign = _fold0(store).model_copy(update={"store": "somewhere_else"})
    with pytest.raises(SplitError, match="was built for"):
        resolve(SignalExamples(store, index), foreign)


def test_resolve_rejects_a_stale_store_digest(store: SignalStore, index) -> None:
    """A split computed against different data must not silently apply to new data."""
    stale = _fold0(store).model_copy(update={"store_manifest_sha256": "0" * 64})
    with pytest.raises(SplitError, match="regenerate the split"):
        resolve(SignalExamples(store, index), stale)


def test_resolve_rejects_unassigned_groups(store: SignalStore, index) -> None:
    """Silently dropping windows is how a fold trains on less data than it claims."""
    partial = SplitFile(
        store=store.path.name, name="partial", parts={"train": ["p0"], "test": ["p1"]}
    )
    with pytest.raises(SplitError, match="does not assign"):
        resolve(SignalExamples(store, index), partial)


def test_resolve_rejects_unknown_groups(store: SignalStore, index) -> None:
    bogus = SplitFile(
        store=store.path.name,
        name="bogus",
        parts={"train": sorted(store.groups), "test": ["ghost"]},
    )
    with pytest.raises(SplitError, match="absent from the index"):
        resolve(SignalExamples(store, index), bogus)
