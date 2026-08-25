"""Split invariants. The leakage tests here are the point of the whole module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import SignalExamples, entity_examples
from dsio.data.store import SignalStore
from dsio.data.views import WindowSpec, build_index
from dsio.splits.models import SCHEMA, SplitError, SplitFile, SplitFold
from dsio.splits.resolve import assert_no_row_overlap, resolve


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
        folds=[
            SplitFold(
                index=0,
                counts={"train": 6, "val": 1, "test": 2},
                parts={
                    "train": ["p2", "p3", "p4", "p5", "p6", "p7"],
                    "val": ["p8"],
                    "test": ["p0", "p1"],
                },
            )
        ],
    )


# --- the check that matters most ----------------------------------------------------


def test_overlapping_parts_are_rejected() -> None:
    """Validating duplicates *within* a part is the obvious check and the insufficient one.

    A group in both train and test passes it and silently invalidates every number the
    split produces.
    """
    with pytest.raises(ValueError, match="mutually disjoint"):
        SplitFile(
            store="s",
            name="bad",
            folds=[SplitFold(index=0, parts={"train": ["p1", "p2"], "test": ["p2", "p3"]})],
        )


def test_duplicates_within_a_part_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        SplitFile(store="s", name="bad", folds=[SplitFold(index=0, parts={"train": ["p1", "p1"]})])


def test_split_with_neither_groups_nor_time_is_rejected() -> None:
    """A fold must divide something; empty is not a valid partition."""
    with pytest.raises(ValueError, match="group parts, temporal bounds, or both"):
        SplitFile(store="s", name="bad", folds=[SplitFold(index=0, parts={})])


def test_split_with_no_folds_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        SplitFile(store="s", name="bad", folds=[])


def test_duplicate_fold_indices_are_rejected() -> None:
    """Two folds declaring the same index make `SplitFile.fold(index)` ambiguous."""
    with pytest.raises(ValueError, match="unique"):
        SplitFile(
            store="s",
            name="bad",
            folds=[
                SplitFold(index=0, parts={"train": ["p1"], "test": ["p2"]}),
                SplitFold(index=0, parts={"train": ["p3"], "test": ["p4"]}),
            ],
        )


def test_no_raw_row_appears_in_two_parts(store: SignalStore, index) -> None:
    """The structural guarantee, verified directly rather than assumed."""
    split = _fold0(store)
    assert_no_row_overlap(resolve(SignalExamples(store, index), split, split.fold(0)))


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
        folds=[
            SplitFold(
                index=0,
                counts={"train": 7, "test": 2},
                parts={"train": sorted(f"p{i}" for i in range(2, 9)), "test": ["p0", "p1"]},
            )
        ],
    )
    path = tmp_path / "splits" / "k3" / "fold0.yaml"
    split.save(path)
    restored = SplitFile.load(path)
    assert restored.folds[0].parts == split.folds[0].parts


def test_foreign_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("schema_version: other/9\nstore: s\nname: n\nparts: {train: [a]}\n")
    with pytest.raises(SplitError, match="expected"):
        SplitFile.load(path)


def test_to_yaml_header_states_counts_and_group_key() -> None:
    """The header is read in a diff without parsing the body, so these two lines — what
    the leakage boundary is, and how big each part is — must actually be there. A
    single-fold family (the common case) gets the plain, unprefixed line."""
    split = SplitFile(
        store="s",
        name="k3",
        group_key="subject",
        folds=[
            SplitFold(
                index=0,
                counts={"train": 2, "test": 1},
                parts={"train": ["a", "b"], "test": ["c"]},
            )
        ],
    )
    header = split.to_yaml()
    assert "# group key: subject  <- the leakage boundary" in header
    assert "# counts: test=1, train=2" in header


# --- fold identity: declared index, not list position --------------------------------


def test_fold_indices_survive_a_yaml_round_trip_undisturbed(tmp_path: Path) -> None:
    """Folds declaring 0, 2, 5 must come back as 0, 2, 5 -- not renumbered 0, 1, 2.

    Nothing here resolves against a store: this is purely about what `SplitFile.load`
    hands back, so a gap in the numbering (fold 1, 3, 4 never existed) is exactly as valid
    as a contiguous run.
    """
    split = SplitFile(
        store="s",
        name="sparse",
        folds=[
            SplitFold(index=0, parts={"train": ["a"], "test": ["b"]}),
            SplitFold(index=2, parts={"train": ["c"], "test": ["d"]}),
            SplitFold(index=5, parts={"train": ["e"], "test": ["f"]}),
        ],
    )
    path = tmp_path / "sparse.yaml"
    split.save(path)
    restored = SplitFile.load(path)
    assert [f.index for f in restored.folds] == [0, 2, 5]
    assert restored.fold(5).parts == {"train": ["e"], "test": ["f"]}
    with pytest.raises(SplitError, match="no fold 1"):
        restored.fold(1)


# --- cross-fold test disjointness: checked at load, before any store is touched -------


def test_load_rejects_a_shared_test_group_across_folds(tmp_path: Path) -> None:
    """Hand-written YAML, not built through `SplitFile(...)` — fold 1 and fold 3 both test
    group 'p2'. Nothing here constructs an `Examples`, opens a store, or resolves
    anything: this proves the check fires on `SplitFile.load` alone, strictly before any
    of that could happen.
    """
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""\
schema_version: {SCHEMA}
store: cohort
name: k4
folds:
  - index: 0
    parts:
      train: [p0]
      test: [p1]
  - index: 1
    parts:
      train: [p3]
      test: [p2]
  - index: 3
    parts:
      train: [p4]
      test: [p2]
"""
    )
    with pytest.raises(SplitError, match=r"folds \[1, 3\].*'p2'"):
        SplitFile.load(path)


def test_load_accepts_disjoint_test_parts_across_folds(tmp_path: Path) -> None:
    """The guard must not fire on a family that is actually fine."""
    path = tmp_path / "good.yaml"
    path.write_text(
        f"""\
schema_version: {SCHEMA}
store: cohort
name: k4
folds:
  - index: 0
    parts:
      train: [p1]
      test: [p0]
  - index: 1
    parts:
      train: [p0]
      test: [p1]
"""
    )
    restored = SplitFile.load(path)
    assert [f.index for f in restored.folds] == [0, 1]


# --- the same checks, on every construction path -- not only `SplitFile.load` --------
#
# The four-cell matrix this section proves:
#
#                          duplicate index      cross-fold test overlap
#   SplitFile.load()          SplitError           SplitError
#   direct construction       ValueError            ValueError
#
# (`SplitError` is itself a `ValueError`, so a `SplitError` also satisfies
# `pytest.raises(ValueError, ...)` -- the two rows are distinguished by asserting the
# concrete exception type where it matters, not merely that *something* was raised.)


def test_overlapping_test_parts_are_rejected_on_direct_construction_too() -> None:
    """This guard used to run only inside `SplitFile.load`; direct construction let a
    cross-fold group collision through with no error at all -- one of the two things
    Fix round 1 closed. It must now reject on `SplitFile(...)` exactly like
    `_validate_parts` and the duplicate-index check already do, and it must NOT be a
    `SplitError` here (that type is reserved for the `SplitFile.load` path)."""
    with pytest.raises(ValueError, match=r"folds \[0, 1\].*'g1'") as excinfo:
        SplitFile(
            store="s",
            name="fam",
            folds=[
                SplitFold(index=0, parts={"train": ["g0"], "test": ["g1"]}),
                SplitFold(index=1, parts={"train": ["g2"], "test": ["g1"]}),
            ],
        )
    assert not isinstance(excinfo.value, SplitError)


def test_load_reports_a_duplicate_fold_index_as_split_error(tmp_path: Path) -> None:
    """Before Fix round 1, `SplitFile.load` called `model_validate` directly, so a
    duplicate fold index leaked a raw `pydantic.ValidationError` out of `load()` instead
    of `SplitError` -- the second thing that fix closed. Hand-written YAML: a
    `SplitFile(...)` built directly with a duplicate index can no longer even be
    constructed (see `test_duplicate_fold_indices_are_rejected`), so there is no object to
    `.save()`.
    """
    path = tmp_path / "dup.yaml"
    path.write_text(
        f"""\
schema_version: {SCHEMA}
store: cohort
name: k5
folds:
  - index: 0
    parts:
      train: [p0]
      test: [p1]
  - index: 0
    parts:
      train: [p2]
      test: [p3]
"""
    )
    with pytest.raises(SplitError, match="unique"):
        SplitFile.load(path)


# --- resolution ---------------------------------------------------------------------


def test_resolve_produces_disjoint_group_sets(store: SignalStore, index) -> None:
    split = _fold0(store)
    parts = resolve(SignalExamples(store, index), split, split.fold(0))
    seen: set[str] = set()
    for sub in parts.values():
        groups = set(sub.groups.tolist())
        assert not (groups & seen)
        seen |= groups


def test_resolve_accounts_for_every_window(store: SignalStore, index) -> None:
    split = _fold0(store)
    parts = resolve(SignalExamples(store, index), split, split.fold(0))
    assert sum(len(sub) for sub in parts.values()) == len(index)


def test_resolve_rejects_a_split_from_another_store(store: SignalStore, index) -> None:
    split = _fold0(store)
    foreign = split.model_copy(update={"store": "somewhere_else"})
    with pytest.raises(SplitError, match="was built for"):
        resolve(SignalExamples(store, index), foreign, foreign.fold(0))


def test_resolve_rejects_a_stale_store_digest(store: SignalStore, index) -> None:
    """A split computed against different data must not silently apply to new data."""
    split = _fold0(store)
    stale = split.model_copy(update={"store_manifest_sha256": "0" * 64})
    with pytest.raises(SplitError, match="regenerate the split"):
        resolve(SignalExamples(store, index), stale, stale.fold(0))


def test_resolve_rejects_unassigned_groups(store: SignalStore, index) -> None:
    """Silently dropping windows is how a fold trains on less data than it claims."""
    partial = SplitFile(
        store=store.path.name,
        name="partial",
        folds=[SplitFold(index=0, parts={"train": ["p0"], "test": ["p1"]})],
    )
    with pytest.raises(SplitError, match="does not assign"):
        resolve(SignalExamples(store, index), partial, partial.fold(0))


def test_resolve_rejects_unknown_groups(store: SignalStore, index) -> None:
    bogus = SplitFile(
        store=store.path.name,
        name="bogus",
        folds=[SplitFold(index=0, parts={"train": sorted(store.groups), "test": ["ghost"]})],
    )
    with pytest.raises(SplitError, match="absent from the index"):
        resolve(SignalExamples(store, index), bogus, bogus.fold(0))
