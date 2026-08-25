"""The seam: committed split YAMLs driving the fold loop over a memory-mapped store.

Everything either side of this was tested in isolation before this file existed — the
store, the window index, the split files, the loop. This is where the design is actually
load-bearing, and the property that matters is that no window a fold trains on can appear
in the window it is scored against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import SignalExamples, entity_examples
from dsio.data.store import SignalStore
from dsio.data.views import WindowSpec, build_index
from dsio.eval.contract import Fold
from dsio.splits.folds import (
    _assert_test_parts_are_disjoint,
    folds_from_splits,
    load_folds,
    require_fold,
    split_path,
)
from dsio.splits.models import SplitError, SplitFile, SplitFold
from dsio.splits.temporal import TemporalSpec, describe, walk_forward


@pytest.fixture
def store(tmp_path: Path) -> SignalStore:
    """Nine groups, two recordings each — the shape a grouped corpus usually takes."""
    path = tmp_path / "cohort"
    rng = np.random.default_rng(0)
    with SignalStore.builder(path, channels=3) as builder:
        for group in range(9):
            for session in range(2):
                builder.add(
                    f"p{group}_s{session}",
                    rng.standard_normal((1500, 3)).astype("float32"),
                    group=f"p{group}",
                    attrs={"t_start": 0.0, "sample_rate": 100.0, "events": group * 10},
                )
    return SignalStore(path)


@pytest.fixture
def index(store: SignalStore):
    return build_index(store, WindowSpec(length=500, stride=200))


def _kfold3(store: SignalStore) -> list[SplitFile]:
    """Three hand-picked folds over the store's nine groups.

    Test parts partition all nine groups exactly once across the three folds — the shape
    the "every window tested once" tests below need. Fold 0 also carries a validation part,
    for the one test that needs one.
    """
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1", "p2"], "val": ["p8"], "train": ["p3", "p4", "p5", "p6", "p7"]},
        {"test": ["p3", "p4", "p5"], "train": ["p0", "p1", "p2", "p6", "p7", "p8"]},
        {"test": ["p6", "p7", "p8"], "train": ["p0", "p1", "p2", "p3", "p4", "p5"]},
    ]
    return [
        SplitFile(
            store=store.path.name,
            store_manifest_sha256=digest,
            name="k3",
            folds=[
                SplitFold(
                    index=i,
                    counts={part: len(members) for part, members in parts.items()},
                    parts=parts,
                )
            ],
        )
        for i, parts in enumerate(folds)
    ]


def _logo(store: SignalStore, *, name: str) -> list[SplitFile]:
    """One fold per group, holding it out — the trivial leave-one-group-out shape."""
    groups = sorted(store.groups)
    digest = entity_examples(store).digest
    return [
        SplitFile(
            store=store.path.name,
            store_manifest_sha256=digest,
            name=name,
            folds=[
                SplitFold(
                    index=i,
                    counts={"train": len(groups) - 1, "test": 1},
                    parts={"train": [g for g in groups if g != held], "test": [held]},
                )
            ],
        )
        for i, held in enumerate(groups)
    ]


def _temporal_folds(
    examples: SignalExamples, spec: TemporalSpec, *, name: str
) -> list[SplitFile]:
    """One walk-forward fold per :func:`walk_forward` bound — calling the two functions
    this ADR keeps, not reimplementing them."""
    t_start, t_end = examples.times()
    return [
        SplitFile(
            store=examples.name,
            store_manifest_sha256=examples.digest,
            name=name,
            folds=[
                SplitFold(index=fold, counts=describe(bounds, t_start, t_end), temporal=bounds)
            ],
        )
        for fold, bounds in enumerate(walk_forward(t_start, t_end, spec))
    ]


# --- building folds -----------------------------------------------------------------


def test_a_split_per_fold_becomes_a_fold_per_split(store: SignalStore, index) -> None:
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    assert len(folds) == 3
    assert [fold.index for fold in folds] == [0, 1, 2]


def _kfold3_as_one_file(store: SignalStore) -> SplitFile:
    """The same three folds as `_kfold3`, held in **one** file — the shape this task adds:
    a whole split family, one committed YAML, folds ordered inside it."""
    digest = entity_examples(store).digest
    folds = [
        {"test": ["p0", "p1", "p2"], "val": ["p8"], "train": ["p3", "p4", "p5", "p6", "p7"]},
        {"test": ["p3", "p4", "p5"], "train": ["p0", "p1", "p2", "p6", "p7", "p8"]},
        {"test": ["p6", "p7", "p8"], "train": ["p0", "p1", "p2", "p3", "p4", "p5"]},
    ]
    return SplitFile(
        store=store.path.name,
        store_manifest_sha256=digest,
        name="k3one",
        folds=[
            SplitFold(
                index=i,
                counts={part: len(members) for part, members in parts.items()},
                parts=parts,
            )
            for i, parts in enumerate(folds)
        ],
    )


def test_one_file_holding_every_fold_builds_the_same_folds_as_one_file_per_fold(
    store: SignalStore, index
) -> None:
    """A split family expressed as one file with an ordered `folds` list drives
    `folds_from_splits` identically to the old one-file-per-fold shape."""
    single_file = folds_from_splits(SignalExamples(store, index), [_kfold3_as_one_file(store)])
    many_files = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    assert [f.index for f in single_file] == [f.index for f in many_files] == [0, 1, 2]
    for a, b in zip(single_file, many_files, strict=True):
        assert np.array_equal(a.train, b.train)
        assert np.array_equal(a.test, b.test)


def test_one_file_round_trips_through_load_and_still_drives_the_loop(
    store: SignalStore, index, tmp_path: Path
) -> None:
    path = tmp_path / "splits" / "k3one" / "split.yaml"
    _kfold3_as_one_file(store).save(path)
    restored = SplitFile.load(path)
    folds = folds_from_splits(SignalExamples(store, index), [restored])
    assert [f.index for f in folds] == [0, 1, 2]
    assert sum(fold.test.size for fold in folds) == len(index)


def test_folds_cover_the_index_without_copying_it(store: SignalStore, index) -> None:
    """Positions into one index, not three datasets. This is the whole point of the design."""
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    for fold in folds:
        assert fold.train.max() < len(index)
        assert fold.test.max() < len(index)
    assert sum(fold.test.size for fold in folds) == len(index)


def test_no_group_is_trained_on_and_tested_in_the_same_fold(store: SignalStore, index) -> None:
    """The leakage property, checked at the level a metric would actually be corrupted."""
    groups = index.groups
    for fold in folds_from_splits(SignalExamples(store, index), _kfold3(store)):
        assert not (set(groups[fold.train]) & set(groups[fold.test]))


def test_no_raw_row_is_shared_between_a_fold_train_and_test(
    store: SignalStore, index
) -> None:
    """Overlapping windows straddling a boundary are the failure this whole layer exists
    to prevent — near-identical rows in train and test simultaneously."""
    for fold in folds_from_splits(SignalExamples(store, index), _kfold3(store)):
        train_rows = index.subset(_mask(len(index), fold.train)).covered_rows()
        test_rows = index.subset(_mask(len(index), fold.test)).covered_rows()
        assert np.intersect1d(train_rows, test_rows).size == 0


def test_every_window_is_tested_exactly_once_across_folds(store: SignalStore, index) -> None:
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    tested = np.concatenate([fold.test for fold in folds])
    assert sorted(tested.tolist()) == list(range(len(index)))


def test_leave_one_group_out_produces_one_fold_per_subject(
    store: SignalStore, index
) -> None:
    splits = _logo(store, name="logo")
    folds = folds_from_splits(SignalExamples(store, index), splits)
    assert len(folds) == 9
    assert all(len(set(index.groups[fold.test])) == 1 for fold in folds)


def test_fold_numbers_come_from_the_file_not_the_list_position(
    store: SignalStore, index
) -> None:
    """Running folds 1 and 2 alone must not renumber them 0 and 1.

    Otherwise `fold0` in an artifact means a different fold depending on which subset was
    run, and two runs of the same experiment stop being comparable.
    """
    splits = _kfold3(store)
    folds = folds_from_splits(SignalExamples(store, index), splits[1:])
    assert [fold.index for fold in folds] == [1, 2]


def test_a_validation_part_is_carried_through(store: SignalStore, index) -> None:
    splits = _kfold3(store)
    fold = folds_from_splits(SignalExamples(store, index), splits)[0]
    assert fold.val is not None and fold.val.size > 0
    assert np.intersect1d(fold.val, fold.test).size == 0


# --- refusals -----------------------------------------------------------------------


def test_overlapping_test_parts_across_folds_are_rejected(store: SignalStore, index) -> None:
    """Each file is individually valid; only comparing them reveals the double-count.

    This exercises `folds_from_splits`'s own resolved-position guard, not the new
    load-time one in `SplitFile.load`: these two `SplitFile` objects are constructed
    directly (never loaded from YAML), and the load-time check only ever sees the folds
    declared *inside one file* — it cannot see an overlap that only exists across two
    separately built `SplitFile` objects. That is the concrete case the resolved-position
    guard still catches on its own.
    """
    splits = _kfold3(store)
    relabeled = splits[0].model_copy(
        update={"folds": [splits[0].folds[0].model_copy(update={"index": 1})]}
    )
    duplicated = [splits[0], relabeled]
    with pytest.raises(SplitError, match=r"k3\[0\].*k3\[1\]"):
        folds_from_splits(SignalExamples(store, index), duplicated)


def test_a_split_without_a_test_part_is_rejected(store: SignalStore, index) -> None:
    splits = _kfold3(store)
    fold0 = splits[0].folds[0]
    partial_fold = fold0.model_copy(update={"parts": {"train": sorted(fold0.all_groups)}})
    partial = splits[0].model_copy(update={"folds": [partial_fold]})
    with pytest.raises(SplitError, match="no 'test' part"):
        folds_from_splits(SignalExamples(store, index), [partial])


def test_no_splits_is_rejected(store: SignalStore, index) -> None:
    with pytest.raises(SplitError, match="no split files"):
        folds_from_splits(SignalExamples(store, index), [])


# --- from disk ----------------------------------------------------------------------


def test_folds_load_from_a_committed_file(store: SignalStore, index, tmp_path: Path) -> None:
    root = tmp_path / "splits"
    _kfold3_as_one_file(store).model_copy(update={"name": "k3"}).save(root / "k3" / "split.yaml")
    folds = load_folds(SignalExamples(store, index), split_path(root, "k3"))
    assert len(folds) == 3
    assert sum(fold.test.size for fold in folds) == len(index)


def test_missing_split_files_say_how_to_make_them(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="commit a split file"):
        split_path(tmp_path, "nothing")


def test_require_fold_finds_a_fold_declared_in_a_single_file_family(
    store: SignalStore, tmp_path: Path
) -> None:
    """The only layout dsio reads: one file holds the whole family."""
    root = tmp_path / "splits"
    _kfold3_as_one_file(store).save(root / "k3one" / "split.yaml")
    found = require_fold(root, "k3one", 1)
    assert found.index == 1


def test_require_fold_names_every_fold_the_family_declares(
    store: SignalStore, tmp_path: Path
) -> None:
    """The message is `SplitFile.fold`'s own, naming every fold the family declares."""
    root = tmp_path / "splits"
    _kfold3_as_one_file(store).model_copy(update={"name": "k3"}).save(root / "k3" / "split.yaml")
    with pytest.raises(SplitError, match=r"split 'k3' has no fold 9; it defines folds"):
        require_fold(root, "k3", 9)


# --- partial coverage from a purged split ---------------------------------------------
#
# There used to be a companion test here, `test_split_files_drive_the_loop_end_to_end`,
# that ran `_kfold3`'s folds through the deleted `cross_validate` loop and re-checked full
# coverage on the resulting `OutOfFold`. That was always a restatement of what
# `test_folds_cover_the_index_without_copying_it` and
# `test_every_window_is_tested_exactly_once_across_folds` above already prove directly on
# the folds themselves, so it is gone rather than rehomed.


def test_a_purged_walk_forward_produces_partial_coverage(store: SignalStore, index) -> None:
    """The discarded band is the point of purging: `require_total=False` is what lets a
    split say so instead of being rejected as incomplete, and the folds it returns must
    cover less than the full index without being empty."""
    splits = _temporal_folds(
        SignalExamples(store, index),
        TemporalSpec(n_splits=2, test_fraction=0.4, label_horizon=100, embargo=100),
        name="wf",
    )
    folds = folds_from_splits(SignalExamples(store, index), splits, require_total=False)
    covered = sum(fold.test.size for fold in folds)
    assert 0 < covered < len(index)


def _mask(size: int, positions: np.ndarray) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[positions] = True
    return mask


# --- disjointness check, vectorised ---------------------------------------------------


def test_overlapping_test_parts_are_rejected() -> None:
    a = Fold(index=0, train=np.array([2, 3]), test=np.array([0, 1]), val=None, name="a")
    b = Fold(index=1, train=np.array([3]), test=np.array([1, 2]), val=None, name="b")
    with pytest.raises(SplitError, match="disjoint"):
        _assert_test_parts_are_disjoint([a, b])


def test_disjoint_test_parts_pass() -> None:
    a = Fold(index=0, train=np.array([2, 3]), test=np.array([0, 1]), val=None, name="a")
    b = Fold(index=1, train=np.array([0, 1]), test=np.array([2, 3]), val=None, name="b")
    _assert_test_parts_are_disjoint([a, b])


# NOTE: `Fold.__post_init__` (eval/contract.py) already rejects a fold whose own
# train and test overlap. Every fold constructed here must be internally valid, or
# the test fails in the constructor and never reaches the function under test.


def test_large_fold_set_is_fast() -> None:
    # train=[2_000_000] is one past the highest test position used below (10 folds of
    # 200_000 each), so every fold stays internally disjoint (Fold.__post_init__) without
    # relying on a negative index — real folds index real arrays, and a negative sentinel
    # would silently wrap to the last row rather than raise if this pattern were copied
    # somewhere that actually indexed with `.train`.
    folds = [
        Fold(
            index=i,
            train=np.array([2_000_000]),
            test=np.arange(i * 200_000, (i + 1) * 200_000),
            val=None,
            name=f"f{i}",
        )
        for i in range(10)
    ]
    _assert_test_parts_are_disjoint(folds)
