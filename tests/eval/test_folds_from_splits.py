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
from dsio.eval.contract import Fold, FoldPrediction
from dsio.eval.loop import cross_validate
from dsio.splits.folds import (
    _assert_test_parts_are_disjoint,
    fold_paths,
    folds_from_splits,
    load_folds,
)
from dsio.splits.models import SplitError, SplitFile
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
            fold=i,
            counts={part: len(members) for part, members in parts.items()},
            parts=parts,
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
            fold=i,
            counts={"train": len(groups) - 1, "test": 1},
            parts={"train": [g for g in groups if g != held], "test": [held]},
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
            fold=fold,
            counts=describe(bounds, t_start, t_end),
            temporal=bounds,
        )
        for fold, bounds in enumerate(walk_forward(t_start, t_end, spec))
    ]


# --- building folds -----------------------------------------------------------------


def test_a_split_per_fold_becomes_a_fold_per_split(store: SignalStore, index) -> None:
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    assert len(folds) == 3
    assert [fold.index for fold in folds] == [0, 1, 2]


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
    """Each file is individually valid; only comparing them reveals the double-count."""
    splits = _kfold3(store)
    duplicated = [splits[0], splits[0].model_copy(update={"fold": 1})]
    with pytest.raises(SplitError, match=r"k3\[0\].*k3\[1\]"):
        folds_from_splits(SignalExamples(store, index), duplicated)


def test_a_split_without_a_test_part_is_rejected(store: SignalStore, index) -> None:
    splits = _kfold3(store)
    partial = splits[0].model_copy(
        update={"parts": {"train": sorted(splits[0].all_groups)}}
    )
    with pytest.raises(SplitError, match="no 'test' part"):
        folds_from_splits(SignalExamples(store, index), [partial])


def test_no_splits_is_rejected(store: SignalStore, index) -> None:
    with pytest.raises(SplitError, match="no split files"):
        folds_from_splits(SignalExamples(store, index), [])


# --- from disk ----------------------------------------------------------------------


def test_folds_load_from_committed_files(store: SignalStore, index, tmp_path: Path) -> None:
    root = tmp_path / "splits"
    for split in _kfold3(store):
        split.save(root / "k3" / f"fold{split.fold}.yaml")
    folds = load_folds(SignalExamples(store, index), fold_paths(root, "k3"))
    assert len(folds) == 3
    assert sum(fold.test.size for fold in folds) == len(index)


def test_fold_paths_order_numerically_not_lexically(store: SignalStore, tmp_path: Path) -> None:
    """Ten folds must not come back as 0, 1, 10, 2 — every fold is individually valid, so
    nothing downstream can detect the permutation."""
    root = tmp_path / "splits"
    for split in _logo(store, name="k9"):
        split.save(root / "k9" / f"fold{split.fold}.yaml")
    (root / "k9" / "fold10.yaml").write_bytes((root / "k9" / "fold0.yaml").read_bytes())
    ordinals = [int(path.stem.removeprefix("fold")) for path in fold_paths(root, "k9")]
    assert ordinals == sorted(ordinals)
    assert ordinals[-1] == 10


def test_missing_split_files_say_how_to_make_them(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="commit a split file"):
        fold_paths(tmp_path, "nothing")


# --- end to end through the loop ----------------------------------------------------


def test_split_files_drive_the_loop_end_to_end(store: SignalStore, index) -> None:
    """A group-level label, cross-validated over the store without materialising a fold."""
    folds = folds_from_splits(SignalExamples(store, index), _kfold3(store))
    labels = (np.array([int(g[1:]) for g in index.groups]) % 2).astype(int)

    def fit_predict(fold: Fold) -> FoldPrediction:
        # A group-mean baseline; the point is the plumbing, not the model.
        prior = float(labels[fold.train].mean())
        held = labels[fold.test]
        score = np.full(held.size, prior)
        return FoldPrediction(y_true=held, y_pred=(score > 0.5).astype(int), y_score=score)

    report, oof = cross_validate(folds, fit_predict, metrics=["accuracy"], n_rows=len(index))
    assert report.coverage == 1.0
    assert len(oof) == len(index)
    assert sorted(oof.row_id.tolist()) == list(range(len(index)))


def test_a_purged_walk_forward_reports_partial_coverage(store: SignalStore, index) -> None:
    """The discarded band is the point of purging, so coverage below 1.0 is correct here —
    and must be recorded rather than silently rounded away."""
    splits = _temporal_folds(
        SignalExamples(store, index),
        TemporalSpec(n_splits=2, test_fraction=0.4, label_horizon=100, embargo=100),
        name="wf",
    )
    folds = folds_from_splits(SignalExamples(store, index), splits, require_total=False)

    def fit_predict(fold: Fold) -> FoldPrediction:
        held = np.zeros(fold.test.size, dtype=int)
        return FoldPrediction(y_true=held, y_pred=held)

    report, _ = cross_validate(folds, fit_predict, metrics=["accuracy"], n_rows=len(index))
    assert 0.0 < report.coverage < 1.0


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
