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

from dsio.data import SignalExamples, SignalStore, WindowSpec, build_index, entity_examples
from dsio.eval import Fold, FoldPrediction, cross_validate
from dsio.splits import (
    SplitError,
    TemporalSpec,
    fold_paths,
    folds_from_splits,
    load_folds,
)
from splitgen import kfold_split_files, logo_split_files, temporal_split_files, write_split_files


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


# --- building folds -----------------------------------------------------------------


def test_a_split_per_fold_becomes_a_fold_per_split(store: SignalStore, index) -> None:
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    folds = folds_from_splits(SignalExamples(store, index), splits)
    assert len(folds) == 3
    assert [fold.index for fold in folds] == [0, 1, 2]


def test_folds_cover_the_index_without_copying_it(store: SignalStore, index) -> None:
    """Positions into one index, not three datasets. This is the whole point of the design."""
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    folds = folds_from_splits(SignalExamples(store, index), splits)
    for fold in folds:
        assert fold.train.max() < len(index)
        assert fold.test.max() < len(index)
    assert sum(fold.test.size for fold in folds) == len(index)


def test_no_group_is_trained_on_and_tested_in_the_same_fold(store: SignalStore, index) -> None:
    """The leakage property, checked at the level a metric would actually be corrupted."""
    groups = index.groups
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    for fold in folds_from_splits(SignalExamples(store, index), splits):
        assert not (set(groups[fold.train]) & set(groups[fold.test]))


def test_no_raw_row_is_shared_between_a_fold_train_and_test(
    store: SignalStore, index
) -> None:
    """Overlapping windows straddling a boundary are the failure this whole layer exists
    to prevent — near-identical rows in train and test simultaneously."""
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    for fold in folds_from_splits(SignalExamples(store, index), splits):
        train_rows = index.subset(_mask(len(index), fold.train)).covered_rows()
        test_rows = index.subset(_mask(len(index), fold.test)).covered_rows()
        assert np.intersect1d(train_rows, test_rows).size == 0


def test_every_window_is_tested_exactly_once_across_folds(store: SignalStore, index) -> None:
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    tested = np.concatenate([fold.test for fold in folds_from_splits(SignalExamples(store,
        index), splits)])
    assert sorted(tested.tolist()) == list(range(len(index)))


def test_leave_one_group_out_produces_one_fold_per_subject(
    store: SignalStore, index
) -> None:
    splits = logo_split_files(entity_examples(store), name="logo")
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
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    folds = folds_from_splits(SignalExamples(store, index), splits[1:])
    assert [fold.index for fold in folds] == [1, 2]


def test_a_validation_part_is_carried_through(store: SignalStore, index) -> None:
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    fold = folds_from_splits(SignalExamples(store, index), splits)[0]
    assert fold.val is not None and fold.val.size > 0
    assert np.intersect1d(fold.val, fold.test).size == 0


# --- refusals -----------------------------------------------------------------------


def test_overlapping_test_parts_across_folds_are_rejected(store: SignalStore, index) -> None:
    """Each file is individually valid; only comparing them reveals the double-count."""
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    duplicated = [splits[0], splits[0].model_copy(update={"fold": 1})]
    with pytest.raises(SplitError, match="test part of both"):
        folds_from_splits(SignalExamples(store, index), duplicated)


def test_a_split_without_a_test_part_is_rejected(store: SignalStore, index) -> None:
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
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
    write_split_files(kfold_split_files(entity_examples(store), 3, name="k3"), name="k3", root=root)
    folds = load_folds(SignalExamples(store, index), fold_paths(root, "k3"))
    assert len(folds) == 3
    assert sum(fold.test.size for fold in folds) == len(index)


def test_fold_paths_order_numerically_not_lexically(store: SignalStore, tmp_path: Path) -> None:
    """Ten folds must not come back as 0, 1, 10, 2 — every fold is individually valid, so
    nothing downstream can detect the permutation."""
    root = tmp_path / "splits"
    write_split_files(kfold_split_files(entity_examples(store), 9, name="k9"), name="k9", root=root)
    (root / "k9" / "fold10.yaml").write_bytes((root / "k9" / "fold0.yaml").read_bytes())
    ordinals = [int(path.stem.removeprefix("fold")) for path in fold_paths(root, "k9")]
    assert ordinals == sorted(ordinals)
    assert ordinals[-1] == 10


def test_missing_split_files_say_how_to_make_them(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="dsio splits make"):
        fold_paths(tmp_path, "nothing")


# --- end to end through the loop ----------------------------------------------------


def test_split_files_drive_the_loop_end_to_end(store: SignalStore, index) -> None:
    """A group-level label, cross-validated over the store without materialising a fold."""
    splits = kfold_split_files(entity_examples(store), 3, name="k3")
    folds = folds_from_splits(SignalExamples(store, index), splits)
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
    splits = temporal_split_files(SignalExamples(store, index),
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
